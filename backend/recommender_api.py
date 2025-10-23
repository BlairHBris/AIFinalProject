# recommender_api.py
"""
Render-safe API.
- DOES NOT train on startup.
- Loads a locally-committed checkpoint file (torch_recommender.pt).
- Uses weights_only=False in torch.load to load full checkpoint created by train_torch.py.
  This is less restrictive than the newer default safe-loading; acceptable here because
  you generate the .pt locally and commit it (trusted source).
- Uses float16 compressed feature matrices and batched scoring for fast recommendations.
"""

import os
import threading
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# Config
# -----------------------
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
TAGS_FILE = "tags.csv"
MODEL_FILE = "torch_recommender.pt"
USERS_FILE = "users.csv"
INTERACTIONS_FILE = "user_interactions.csv"
UPDATE_THRESHOLD = 10

# -----------------------
# FastAPI + CORS
# -----------------------
app = FastAPI(title="Movie Recommender (render-safe)")
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

# -----------------------
# Globals (populated on startup)
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
user2idx = {}
movie2idx = {}
content_cols = []
movie_lookup = {}
movies = None
movies_features = None           # DataFrame indexed by movieid with content_cols, dtype float16
movieid2row = {}                 # movieid -> row index in movies_features (fast lookup)

# -----------------------
# Model definition (matching training)
# -----------------------
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, num_features, embedding_dim=32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.movie_embed = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim + num_features, 1)

    def forward(self, user_idx, movie_idx, features):
        u_emb = self.user_embed(user_idx)
        m_emb = self.movie_embed(movie_idx)
        x = u_emb * m_emb
        x = torch.cat([x, features], dim=1)
        x = self.fc(x)
        return x.squeeze()

# -----------------------
# Utilities to load CSVs & feature matrices
# -----------------------
def _load_movies_and_features():
    global movies, movie_lookup, content_cols, movies_features, movieid2row
    if not os.path.exists(MOVIES_FILE):
        raise FileNotFoundError("movies.csv not found in repo (needed on server).")

    movies = pd.read_csv(MOVIES_FILE)
    movies.columns = [c.strip().lower() for c in movies.columns]
    movies["genres"] = movies.get("genres", "").fillna("")
    if "actors" not in movies.columns:
        movies["actors"] = ""

    unique_genres = set("|".join(movies["genres"]).split("|"))
    unique_genres = {g for g in unique_genres if g}
    for g in unique_genres:
        movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0).astype(np.float16)

    # tags: if present, attempt to load and vectorize to same top-K size as training expected.
    tag_cols = []
    if os.path.exists(TAGS_FILE):
        tags = pd.read_csv(TAGS_FILE)
        tags.columns = [c.strip().lower() for c in tags.columns]
        tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
        movie_tags = tags.groupby("movieid")["tag"].apply(lambda x: " ".join(x.unique())).reset_index()
        # We don't know exact vocabulary used during training, but the training script saved content_cols in checkpoint.
        # We'll still vectorize to keep per-movie numeric columns if no checkpoint is present (fallback).
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=512, token_pattern=r"(?u)\b\w+\b")
        tag_mat = vectorizer.fit_transform(movie_tags["tag"].fillna("")).toarray().astype(np.float16)
        tag_cols = vectorizer.get_feature_names_out()
        tag_df = pd.DataFrame(tag_mat, columns=tag_cols, dtype=np.float16)
        tag_df["movieid"] = movie_tags["movieid"].astype(np.int32)
        movies = movies.merge(tag_df, on="movieid", how="left").fillna(0)

    content_cols = sorted(list(unique_genres)) + list(tag_cols)
    movie_lookup = movies.set_index("movieid")["title"].to_dict()

    # precompute features matrix and mapping (movieid -> row index)
    movies_features = movies.set_index("movieid")[content_cols].astype(np.float16)
    movieid2row = {int(mid): i for i, mid in enumerate(movies_features.index)}

    # expose to module-level
    return movies, movie_lookup, list(content_cols), movies_features, movieid2row

# -----------------------
# Load model checkpoint (weights_only=False option)
# -----------------------
def load_model_checkpoint():
    """
    Loads the full checkpoint saved by train_torch.py.
    We use torch.load(..., weights_only=False) to allow loading the dict saved by the training script.
    This is safe because the .pt should be produced locally and trusted.
    """
    global model, user2idx, movie2idx, content_cols, movies, movie_lookup, movies_features, movieid2row

    # ensure movie features are present
    movies, movie_lookup, content_cols, movies_features, movieid2row = _load_movies_and_features()

    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model checkpoint not found. Please run train_torch.py locally and commit torch_recommender.pt.")
        return

    # IMPORTANT: torch.load default changed in PyTorch 2.6 to weights-only semantics.
    # We intentionally instruct torch to load the full pickled checkpoint (weights_only=False).
    # This is only safe if you trust the .pt file (you generated it locally).
    chk = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    # Expect keys: model_state_dict, user2idx, movie2idx, content_cols (training saved these)
    if "model_state_dict" in chk and "user2idx" in chk and "movie2idx" in chk:
        user2idx = chk["user2idx"]
        movie2idx = chk["movie2idx"]

        # Create model with sizes that match the checkpoint maps
        model = RecommenderNet(len(user2idx), len(movie2idx), len(content_cols)).to(device)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        print(f"✅ Loaded checkpoint from {MODEL_FILE} (weights_only=False).")
    else:
        raise RuntimeError("Checkpoint missing required keys. Recreate checkpoint with train_torch.py.")

# -----------------------
# Prediction helpers (batched)
# -----------------------
def predict_batch(user_idx_int, movie_id_batch):
    """Return list of floats score for this user for the movie_id_batch."""
    # Build tensors
    u_idx = user_idx_int
    # map movie ids to movie2idx (if missing, use 0)
    m_idxs = [movie2idx.get(int(mid), 0) for mid in movie_id_batch]

    # Build user tensor repeated
    u = torch.tensor([u_idx] * len(m_idxs), dtype=torch.long, device=device)
    m = torch.tensor(m_idxs, dtype=torch.long, device=device)

    # Get features for batch (fast lookup via movies_features and movieid2row)
    feat_rows = []
    for mid in movie_id_batch:
        if int(mid) in movieid2row:
            feat_rows.append(movies_features.iloc[movieid2row[int(mid)]].values)
        else:
            feat_rows.append(np.zeros((len(content_cols),), dtype=np.float16))
    feat_arr = np.stack(feat_rows).astype(np.float32)   # convert to float32 for stable fc computation
    f = torch.tensor(feat_arr, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(u, m, f)
    return preds.cpu().numpy()

def recommend_for_user(user_id, top_n=10, candidate_limit=500, batch_size=64):
    # map our user_id (from USERS_FILE) to model user index; if user not in user2idx, fallback to 0 (unknown)
    # Note: get_or_create_user maps to app-specific user IDs, but model trained on movielens user ids.
    # Here we'll attempt direct mapping; missing users treat as unknown (embedding 0).
    u_idx = user2idx.get(user_id, 0)

    # exclude already-interacted movies (server-side interactions file)
    seen = set()
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        seen = set(df[df["user_id"] == int(user_id)]["movie_id"].astype(int))

    candidates = [mid for mid in movie_lookup.keys() if mid not in seen]
    if len(candidates) > candidate_limit:
        # random subsample to speed up scoring but keep variety
        candidates = list(np.random.choice(candidates, candidate_limit, replace=False))

    preds = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        scores = predict_batch(u_idx, batch)
        preds.extend(zip(batch, scores))

    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": int(mid), "title": movie_lookup.get(int(mid), "Unknown")} for mid, _ in top_preds]

# -----------------------
# API data models
# -----------------------
class RecommendRequest(BaseModel):
    username: str
    liked_genres: list[str] = []
    liked_actors: list[str] = []
    top_n: int = 5

class FeedbackRequest(BaseModel):
    username: str
    movie_id: int
    interaction: str

# -----------------------
# User bookkeeping
# -----------------------
def get_or_create_user(username: str) -> int:
    username = username.strip().lower()
    if os.path.exists(USERS_FILE):
        users_df = pd.read_csv(USERS_FILE)
    else:
        users_df = pd.DataFrame(columns=["user_id", "username"])
    if username in users_df["username"].values:
        return int(users_df.loc[users_df["username"] == username, "user_id"].iloc[0])
    new_id = int(users_df["user_id"].max() + 1) if not users_df.empty else 1
    users_df = pd.concat([users_df, pd.DataFrame([{"user_id": new_id, "username": username}])], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    return new_id

# -----------------------
# Startup: load features and checkpoint
# -----------------------
@app.on_event("startup")
def startup_event():
    # load feature matrices first
    try:
        _load_movies_and_features()
    except Exception as e:
        print("Failed to load movies/features on startup:", e)
    # load checkpoint (no training)
    try:
        load_model_checkpoint()
    except Exception as e:
        # If loading fails, let server start but endpoints that rely on model will error.
        print("Failed to load model checkpoint:", e)

# -----------------------
# API routes
# -----------------------
@app.post("/recommend/{rec_type}")
def recommend(rec_type: str, req: RecommendRequest):
    user_id = get_or_create_user(req.username)
    # For Render-safe server we ignore rec_type variants and return the fast model-based recommendations.
    recs = recommend_for_user(user_id, top_n=req.top_n)
    return {"recommendations": recs}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id", "movie_id", "interaction"])
    new_row = {"user_id": int(user_id), "movie_id": int(req.movie_id), "interaction": req.interaction}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(INTERACTIONS_FILE, index=False)
    # trigger lightweight background retrain when threshold met
    if len(df) % UPDATE_THRESHOLD == 0:
        threading.Thread(target=background_retrain, daemon=True).start()
    return {"status": "success", "message": "Feedback saved."}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id", "movie_id", "interaction"])
    user_rows = df[df["user_id"] == user_id]
    return {"history": [{"movieId": int(mid), "title": movie_lookup.get(int(mid), "Unknown"), "interaction": inter}
                        for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"])]}

@app.get("/warmup")
def warmup():
    # load checkpoint if not already loaded
    if model is None:
        try:
            load_model_checkpoint()
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    return {"status": "ready", "device": str(device)}