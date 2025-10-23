# recommender_api.py (local training + render optimized + warmup)
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
TMP_MODEL_FILE = "/tmp/torch_recommender.pt"

USERS_FILE = "users.csv"
INTERACTIONS_FILE = "user_interactions.csv"
UPDATE_THRESHOLD = 10

# =======================
# 1. FastAPI setup
# =======================
app = FastAPI(title="Adaptive Movie Recommender API (Local/Render Optimized)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 2. Load local CSVs
# =======================
if not os.path.exists(MOVIES_FILE) or not os.path.exists(RATINGS_FILE):
    raise FileNotFoundError("Missing local CSV files: movies.csv or ratings.csv")

movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)
tags = pd.read_csv(TAGS_FILE) if os.path.exists(TAGS_FILE) else pd.DataFrame()

movies.columns = [c.strip().lower() for c in movies.columns]
ratings.columns = [c.strip().lower() for c in ratings.columns]

movies["genres"] = movies.get("genres", "").fillna("")
if "actors" not in movies.columns:
    movies["actors"] = ""

unique_genres = set("|".join(movies["genres"]).split("|"))
unique_genres = {g for g in unique_genres if g}
for g in unique_genres:
    movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0)

tag_cols = []
if not tags.empty:
    tags.columns = [c.strip().lower() for c in tags.columns]
    tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
    movie_tags = tags.groupby("movieid")["tag"].apply(lambda x: " ".join(x.unique())).reset_index()
    vectorizer = CountVectorizer(max_features=1000, token_pattern=r"(?u)\b\w+\b")
    tag_matrix = vectorizer.fit_transform(movie_tags["tag"].fillna("")).toarray()
    tag_cols = vectorizer.get_feature_names_out()
    tags_df = pd.DataFrame(tag_matrix, columns=tag_cols)
    tags_df["movieid"] = movie_tags["movieid"]
    movies = movies.merge(tags_df, on="movieid", how="left").fillna(0)

content_cols = sorted(list(unique_genres)) + list(tag_cols)
movie_lookup = movies.set_index("movieid")["title"].to_dict()
ratings_full = ratings.merge(movies[["movieid"] + content_cols], on="movieid", how="left")

# =======================
# 3. Model + Dataset
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

user_ids = ratings["userid"].unique()
movie_ids = ratings["movieid"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = len(content_cols)
model = RecommenderNet(len(user2idx), len(movie2idx), num_features).to(device)

class RatingsDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df.copy()
        self.df = self.df[self.df["userid"].isin(user2idx) & self.df["movieid"].isin(movie2idx)].reset_index(drop=True)
        self.users = torch.tensor(self.df["userid"].map(user2idx).values, dtype=torch.long)
        self.movies = torch.tensor(self.df["movieid"].map(movie2idx).values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32)
        self.features = torch.tensor(self.df[feature_cols].fillna(0).values, dtype=torch.float32)

    def __len__(self): return len(self.ratings)
    def __getitem__(self, idx): return self.users[idx], self.movies[idx], self.features[idx], self.ratings[idx]

# =======================
# 4. Load model only (Render-safe)
# =======================
def load_model():
    """Loads pretrained model if available, otherwise a fresh one (no training on Render)."""
    if os.path.exists(MODEL_FILE):
        checkpoint = torch.load(MODEL_FILE, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded pretrained model from {MODEL_FILE}")
    else:
        print("⚠️ Model file not found. Please train locally using train_torch.py and commit the .pt file.")
    model.eval()

# =======================
# 5. Background retrain (Render-safe incremental updates)
# =======================
def background_retrain():
    print("🔁 Running background retrain...")
    df = pd.read_csv(INTERACTIONS_FILE)
    if len(df) < UPDATE_THRESHOLD:
        print("Not enough feedback yet for retrain.")
        return
    # lightweight fine-tune
    dataset = RatingsDataset(ratings_full, content_cols)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1):
        for u, m, f, r in loader:
            u, m, f, r = u.to(device), m.to(device), f.to(device), r.to(device)
            optimizer.zero_grad()
            loss = criterion(model(u, m, f), r)
            loss.backward()
            optimizer.step()
    torch.save({"model_state_dict": model.state_dict()}, MODEL_FILE)
    model.eval()
    print("✅ Background retrain complete and model saved.")

# =======================
# 6. User and Feedback
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
    users_df = pd.read_csv(USERS_FILE) if os.path.exists(USERS_FILE) else pd.DataFrame(columns=["user_id", "username"])
    if username in users_df["username"].values:
        return int(users_df.loc[users_df["username"] == username, "user_id"].iloc[0])
    new_id = int(users_df["user_id"].max() + 1) if not users_df.empty else 1
    users_df = pd.concat([users_df, pd.DataFrame([{"user_id": new_id, "username": username}])], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    return new_id

# =======================
# 7. Recommender logic (fast)
# =======================
def predict_rating(user_id, movie_id):
    if user_id not in user2idx or movie_id not in movie2idx:
        return 3.5
    u = torch.tensor([user2idx[user_id]], dtype=torch.long, device=device)
    m = torch.tensor([movie2idx[movie_id]], dtype=torch.long, device=device)
    feat_vals = movies.loc[movies["movieid"] == movie_id, content_cols].values[0]
    f = torch.tensor([feat_vals], dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(model(u, m, f).item())

def recommend_fast(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = set()
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        rated = set(df[df["user_id"] == user_id]["movie_id"])
    candidates = [mid for mid in movie_lookup.keys() if mid not in rated]
    if len(candidates) > 500:
        candidates = np.random.choice(candidates, 500, replace=False)
    preds = [(mid, predict_rating(user_id, mid)) for mid in candidates]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": int(mid), "title": movie_lookup.get(mid, "Unknown")} for mid, _ in top_preds]

# =======================
# 8. API Routes
# =======================
@app.on_event("startup")
def startup_event():
    load_model()  # ⚙️ Always load, never train on Render

@app.post("/recommend/{rec_type}")
def recommend(rec_type: str, req: RecommendRequest):
    user_id = get_or_create_user(req.username)
    recs = recommend_fast(user_id, req.liked_genres, req.liked_actors, req.top_n)
    return {"recommendations": recs}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id", "movie_id", "interaction"])
    new_row = {"user_id": int(user_id), "movie_id": int(req.movie_id), "interaction": req.interaction}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(INTERACTIONS_FILE, index=False)
    if len(df) % UPDATE_THRESHOLD == 0:
        threading.Thread(target=background_retrain).start()
    return {"status": "success", "message": "Feedback saved."}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id", "movie_id", "interaction"])
    user_rows = df[df["user_id"] == user_id]
    return {"history": [{"movieId": int(mid), "title": movie_lookup.get(mid, "Unknown"), "interaction": inter}
                        for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"])]}

# ⚡ Warmup Route
@app.get("/warmup")
def warmup_model():
    """Preloads model weights and ensures everything is ready."""
    load_model()
    dummy_user = list(user2idx.keys())[0]
    dummy_movie = list(movie2idx.keys())[0]
    _ = predict_rating(dummy_user, dummy_movie)
    return {"status": "ready", "device": str(device)}
