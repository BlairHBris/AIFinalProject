# recommender_api_render_safe_v5.py
import os
import threading
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# =======================
# 0. Config
# =======================
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
TAGS_FILE = "tags.csv"
MODEL_FILE = "torch_recommender.pt"
USERS_FILE = "users.csv"
INTERACTIONS_FILE = "user_interactions.csv"
UPDATE_THRESHOLD = 10

# =======================
# 1. FastAPI setup
# =======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://movie-recommender-frontend-nxw6.onrender.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 2. Globals
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
user2idx, movie2idx = {}, {}
movies, movie_lookup, content_cols = None, None, None
movies_features, movieid2row = None, None

# =======================
# 3. Load movies + tags (memory optimized)
# =======================
def load_movies_and_tags():
    global movies, movie_lookup, content_cols, movies_features, movieid2row

    movies = pd.read_csv(MOVIES_FILE)
    movies.columns = [c.strip().lower() for c in movies.columns]
    movies["genres"] = movies.get("genres", "").fillna("")
    if "actors" not in movies.columns:
        movies["actors"] = ""

    # Genres one-hot (float16)
    unique_genres = set("|".join(movies["genres"]).split("|"))
    unique_genres = {g for g in unique_genres if g}
    for g in unique_genres:
        movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0).astype(np.float16)

    # Tags features
    tag_cols = []
    if os.path.exists(TAGS_FILE):
        tags = pd.read_csv(TAGS_FILE)
        tags.columns = [c.strip().lower() for c in tags.columns]
        tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
        movie_tags = tags.groupby("movieid")["tag"].apply(lambda x: " ".join(x.unique())).reset_index()
        vectorizer = CountVectorizer(max_features=512, token_pattern=r"(?u)\b\w+\b")
        tag_matrix = vectorizer.fit_transform(movie_tags["tag"].fillna("")).toarray().astype(np.float16)
        tag_cols = vectorizer.get_feature_names_out()
        tags_df = pd.DataFrame(tag_matrix, columns=tag_cols, dtype=np.float16)
        tags_df["movieid"] = movie_tags["movieid"].astype(np.int32)
        movies = movies.merge(tags_df, on="movieid", how="left").fillna(0)

    content_cols = sorted(list(unique_genres)) + list(tag_cols)
    movie_lookup = movies.set_index("movieid")["title"].to_dict()

    # Precompute features matrix
    movies_features = movies.set_index("movieid")[content_cols].astype(np.float16)
    movieid2row = {mid: i for i, mid in enumerate(movies_features.index)}

# =======================
# 4. Model + Dataset
# =======================
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, num_features, embedding_dim=32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users+1, embedding_dim, padding_idx=0)
        self.movie_embed = nn.Embedding(num_movies+1, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim + num_features, 1)

    def forward(self, user_idx, movie_idx, features):
        with torch.autocast(device_type='cuda', dtype=torch.float16 if device.type=='cuda' else torch.float32):
            u_emb = self.user_embed(user_idx)
            m_emb = self.movie_embed(movie_idx)
            x = u_emb * m_emb
            x = torch.cat([x, features], dim=1)
            x = self.fc(x)
        return x.squeeze()

class RatingsDataset(Dataset):
    def __init__(self, df, feature_cols, u2i, m2i):
        self.df = df.copy()
        self.df = self.df[self.df["userid"].isin(u2i) & self.df["movieid"].isin(m2i)].reset_index(drop=True)
        self.users = torch.tensor(self.df["userid"].map(u2i).values, dtype=torch.long)
        self.movies = torch.tensor(self.df["movieid"].map(m2i).values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32)
        self.features = torch.tensor(self.df[feature_cols].fillna(0).values, dtype=torch.float16)

    def __len__(self): return len(self.ratings)
    def __getitem__(self, idx): return self.users[idx], self.movies[idx], self.features[idx], self.ratings[idx]

# =======================
# 5. Load pretrained model
# =======================
def load_model():
    global model, user2idx, movie2idx
    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model file not found. Train locally and commit the .pt file.")
        return
    # Safe loading for PyTorch 2.6+
    import torch.serialization
    with torch.serialization.safe_globals([np.float16, np.float32, np.int32, np.int64]):
        checkpoint = torch.load(MODEL_FILE, map_location=device)
    user2idx = checkpoint["user2idx"]
    movie2idx = checkpoint["movie2idx"]
    model = RecommenderNet(len(user2idx), len(movie2idx), len(content_cols)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded pretrained model from {MODEL_FILE}")

# =======================
# 6. Background retrain
# =======================
def background_retrain():
    if not os.path.exists(INTERACTIONS_FILE):
        return
    df = pd.read_csv(INTERACTIONS_FILE)
    if len(df) < UPDATE_THRESHOLD:
        return

    print("🔁 Running background retrain...")
    ratings_full = pd.read_csv(RATINGS_FILE).merge(
        movies[["movieid"] + content_cols], on="movieid", how="left"
    ).astype({c: np.float16 for c in content_cols})

    ratings_full = ratings_full[
        ratings_full["userid"].isin(user2idx) & ratings_full["movieid"].isin(movie2idx)
    ].reset_index(drop=True)

    ratings_full["user_idx"] = ratings_full["userid"].map(user2idx)
    ratings_full["movie_idx"] = ratings_full["movieid"].map(movie2idx)

    dataset = RatingsDataset(ratings_full, content_cols, user2idx, movie2idx)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for u, m, f, r in loader:
        u, m, f, r = u.to(device), m.to(device), f.to(device), r.to(device)
        optimizer.zero_grad()
        pred = model(u, m, f)
        loss = criterion(pred, r)
        loss.backward()
        optimizer.step()

    torch.save({
        "model_state_dict": model.state_dict(),
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "content_cols": content_cols
    }, MODEL_FILE)
    model.eval()
    print("✅ Background retrain complete and model saved.")

# =======================
# 7. API logic
# =======================
class RecommendRequest(BaseModel):
    username: str
    liked_genres: list[str] = []
    liked_actors: list[str] = []
    top_n: int = 5

class FeedbackRequest(BaseModel):
    username: str
    movie_id: int
    interaction: str

def get_or_create_user(username: str) -> int:
    username = username.strip().lower()
    users_df = pd.read_csv(USERS_FILE) if os.path.exists(USERS_FILE) else pd.DataFrame(columns=["user_id","username"])
    if username in users_df["username"].values:
        return int(users_df.loc[users_df["username"]==username,"user_id"].iloc[0])
    new_id = int(users_df["user_id"].max()+1) if not users_df.empty else 1
    users_df = pd.concat([users_df, pd.DataFrame([{"user_id": new_id, "username": username}])], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    return new_id

def predict_rating(user_id, movie_id):
    u_idx = user2idx.get(user_id, 0)
    m_idx = movie2idx.get(movie_id, 0)
    u = torch.tensor([u_idx], dtype=torch.long, device=device)
    m = torch.tensor([m_idx], dtype=torch.long, device=device)
    if movie_id in movieid2row:
        feat_vals = movies_features.iloc[movieid2row[movie_id]].values[None, :]
    else:
        feat_vals = np.zeros((1, len(content_cols)), dtype=np.float16)
    f = torch.tensor(feat_vals, dtype=torch.float16 if device.type=='cuda' else torch.float32, device=device)
    with torch.no_grad():
        return float(model(u, m, f).item())

def recommend_fast(user_id, liked_genres=[], liked_actors=[], top_n=10, batch_size=50):
    rated = set()
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        rated = set(df[df["user_id"]==user_id]["movie_id"])
    candidates = [mid for mid in movie_lookup.keys() if mid not in rated]
    if len(candidates) > 500:
        candidates = np.random.choice(candidates, 500, replace=False)

    preds = []
    u_idx = user2idx.get(user_id, 0)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16 if device.type=='cuda' else torch.float32):
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            u = torch.tensor([u_idx]*len(batch), dtype=torch.long, device=device)
            m = torch.tensor([movie2idx.get(mid,0) for mid in batch], dtype=torch.long, device=device)
            feat_vals = np.array([movies_features.iloc[movieid2row.get(mid,0)].values for mid in batch], dtype=np.float16)
            f = torch.tensor(feat_vals, dtype=torch.float16 if device.type=='cuda' else torch.float32, device=device)
            batch_preds = model(u, m, f)
            preds.extend(zip(batch, batch_preds.cpu().numpy()))

    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": int(mid), "title": movie_lookup.get(mid,"Unknown")} for mid,_ in top_preds]

# =======================
# 8. API routes
# =======================
@app.on_event("startup")
def startup_event():
    load_movies_and_tags()
    load_model()

@app.post("/recommend/{rec_type}")
def recommend(rec_type: str, req: RecommendRequest):
    user_id = get_or_create_user(req.username)
    recs = recommend_fast(user_id, req.liked_genres, req.liked_actors, req.top_n)
    return {"recommendations": recs}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id","movie_id","interaction"])
    new_row = {"user_id": int(user_id),"movie_id": int(req.movie_id),"interaction": req.interaction}
    df = pd.concat([df,pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(INTERACTIONS_FILE, index=False)
    if len(df) % UPDATE_THRESHOLD == 0:
        threading.Thread(target=background_retrain).start()
    return {"status":"success","message":"Feedback saved."}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id","movie_id","interaction"])
    user_rows = df[df["user_id"]==user_id]
    return {"history":[{"movieId": int(mid), "title": movie_lookup.get(mid,"Unknown"), "interaction": inter} for mid,inter in zip(user_rows["movie_id"],user_rows["interaction"])]}

@app.get("/warmup")
def warmup_model():
    if not user2idx or not movie2idx: return {"status":"ready","device": str(device)}
    dummy_user = list(user2idx.keys())[0]
    dummy_movie = list(movie2idx.keys())[0]
    _ = predict_rating(dummy_user,dummy_movie)
    return {"status":"ready","device": str(device)}
