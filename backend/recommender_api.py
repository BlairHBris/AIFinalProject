# recommender_api_render_safe.py
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
# 2. Load movies only (minimal)
# =======================
movies = pd.read_csv(MOVIES_FILE)
movies.columns = [c.strip().lower() for c in movies.columns]
movies["genres"] = movies.get("genres", "").fillna("")
if "actors" not in movies.columns:
    movies["actors"] = ""

unique_genres = set("|".join(movies["genres"]).split("|"))
unique_genres = {g for g in unique_genres if g}
for g in unique_genres:
    movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0)

tag_cols = []
if os.path.exists(TAGS_FILE):
    tags = pd.read_csv(TAGS_FILE)
    tags.columns = [c.strip().lower() for c in tags.columns]
    tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
    movie_tags = tags.groupby("movieid")["tag"].apply(lambda x: " ".join(x.unique())).reset_index()
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1000, token_pattern=r"(?u)\b\w+\b")
    tag_matrix = vectorizer.fit_transform(movie_tags["tag"].fillna("")).toarray().astype(np.float16)
    tag_cols = vectorizer.get_feature_names_out()
    tags_df = pd.DataFrame(tag_matrix, columns=tag_cols)
    tags_df["movieid"] = movie_tags["movieid"]
    movies = movies.merge(tags_df, on="movieid", how="left").fillna(0).astype(np.float16)

content_cols = sorted(list(unique_genres)) + list(tag_cols)
movie_lookup = movies.set_index("movieid")["title"].to_dict()

# =======================
# 3. Model + dataset placeholders
# =======================
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, num_features, embedding_dim=50):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.movie_embed = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim + num_features, 1)

    def forward(self, user_idx, movie_idx, features):
        u_emb = self.user_embed(user_idx)
        m_emb = self.movie_embed(movie_idx)
        x = u_emb * m_emb
        x = torch.cat([x, features], dim=1)
        x = self.fc(x)
        return x.squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
user2idx, movie2idx = {}, {}

# =======================
# 4. Load model only
# =======================
def load_model():
    global model, user2idx, movie2idx
    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model file not found. Train locally and commit the .pt file.")
        return
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    user2idx = checkpoint["user2idx"]
    movie2idx = checkpoint["movie2idx"]
    model = RecommenderNet(len(user2idx), len(movie2idx), len(content_cols)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded pretrained model from {MODEL_FILE}")

# =======================
# 5. Background retrain (Render-safe)
# =======================
class RatingsDataset(Dataset):
    def __init__(self, df, feature_cols, u2i, m2i):
        self.df = df.copy()
        self.df = self.df[self.df["userid"].isin(u2i) & self.df["movieid"].isin(m2i)].reset_index(drop=True)
        self.users = torch.tensor(self.df["userid"].map(u2i).values, dtype=torch.long)
        self.movies = torch.tensor(self.df["movieid"].map(m2i).values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32)
        self.features = torch.tensor(self.df[feature_cols].fillna(0).values, dtype=torch.float32)

    def __len__(self): return len(self.ratings)
    def __getitem__(self, idx): return self.users[idx], self.movies[idx], self.features[idx], self.ratings[idx]

def background_retrain():
    if not os.path.exists(INTERACTIONS_FILE):
        return
    df = pd.read_csv(INTERACTIONS_FILE)
    if len(df) < UPDATE_THRESHOLD: return
    print("🔁 Running background retrain...")
    ratings_full = pd.read_csv(RATINGS_FILE).merge(movies[["movieid"] + content_cols], on="movieid", how="left").astype(np.float16)
    dataset = RatingsDataset(ratings_full, content_cols, user2idx, movie2idx)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for u, m, f, r in loader:
        u, m, f, r = u.to(device), m.to(device), f.to(device), r.to(device)
        optimizer.zero_grad()
        loss = criterion(model(u, m, f), r)
        loss.backward()
        optimizer.step()
    torch.save({"model_state_dict": model.state_dict(), "user2idx": user2idx, "movie2idx": movie2idx}, MODEL_FILE)
    model.eval()
    print("✅ Background retrain complete and model saved.")

# =======================
# 6. API logic
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
    if user_id not in user2idx or movie_id not in movie2idx:
        return 3.5
    u = torch.tensor([user2idx[user_id]], dtype=torch.long, device=device)
    m = torch.tensor([movie2idx[movie_id]], dtype=torch.long, device=device)
    feat_vals = movies.loc[movies["movieid"]==movie_id, content_cols].values[0]
    f = torch.tensor([feat_vals], dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(model(u, m, f).item())

def recommend_fast(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = set()
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        rated = set(df[df["user_id"]==user_id]["movie_id"])
    candidates = [mid for mid in movie_lookup.keys() if mid not in rated]
    if len(candidates) > 500:
        candidates = np.random.choice(candidates, 500, replace=False)
    preds = [(mid, predict_rating(user_id, mid)) for mid in candidates]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": int(mid), "title": movie_lookup.get(mid,"Unknown")} for mid,_ in top_preds]

# =======================
# 7. API routes
# =======================
@app.on_event("startup")
def startup_event():
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
    load_model()
    if not user2idx or not movie2idx: return {"status":"ready","device": str(device)}
    dummy_user = list(user2idx.keys())[0]
    dummy_movie = list(movie2idx.keys())[0]
    _ = predict_rating(dummy_user,dummy_movie)
    return {"status":"ready","device": str(device)}
