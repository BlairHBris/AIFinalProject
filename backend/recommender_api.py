# recommender_api.py
"""
Render-safe API with full PyTorch recommender integration.
- Loads checkpoint (torch_recommender.pt) on startup.
- Uses CORS to allow frontend requests from Render and localhost.
- Handles feedback and user history.
"""

import os
import threading
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
app = FastAPI(title="Movie Recommender (Render-safe)", version="v6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend live", "cors": "enabled"}

# -----------------------
# Globals
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
user2idx = {}
movie2idx = {}
content_cols = []
movie_lookup = {}
movies = None
movies_features = None
movieid2row = {}

# -----------------------
# Model
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
# Utilities
# -----------------------
def _load_movies_and_features():
    global movies, movie_lookup, content_cols, movies_features, movieid2row
    if not os.path.exists(MOVIES_FILE):
        raise FileNotFoundError("movies.csv not found.")
    movies = pd.read_csv(MOVIES_FILE)
    movies.columns = [c.strip().lower() for c in movies.columns]
    movies["genres"] = movies.get("genres", "").fillna("")
    if "actors" not in movies.columns:
        movies["actors"] = ""

    unique_genres = {g for g in "|".join(movies["genres"]).split("|") if g}
    for g in unique_genres:
        movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0).astype(np.float16)

    tag_cols = []
    if os.path.exists(TAGS_FILE):
        tags = pd.read_csv(TAGS_FILE)
        tags.columns = [c.strip().lower() for c in tags.columns]
        tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
        movie_tags = tags.groupby("movieid")["tag"].apply(lambda x: " ".join(x.unique())).reset_index()
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=512, token_pattern=r"(?u)\b\w+\b")
        tag_mat = vectorizer.fit_transform(movie_tags["tag"].fillna("")).toarray().astype(np.float16)
        tag_cols = vectorizer.get_feature_names_out()
        tag_df = pd.DataFrame(tag_mat, columns=tag_cols, dtype=np.float16)
        tag_df["movieid"] = movie_tags["movieid"].astype(np.int32)
        movies = movies.merge(tag_df, on="movieid", how="left").fillna(0)

    content_cols = sorted(list(unique_genres)) + list(tag_cols)
    movie_lookup = movies.set_index("movieid")["title"].to_dict()
    movies_features = movies.set_index("movieid")[content_cols].astype(np.float16)
    movieid2row = {int(mid): i for i, mid in enumerate(movies_features.index)}
    return movies, movie_lookup, content_cols, movies_features, movieid2row

def load_model_checkpoint():
    global model, user2idx, movie2idx, content_cols, movies, movie_lookup, movies_features, movieid2row
    movies, movie_lookup, content_cols, movies_features, movieid2row = _load_movies_and_features()
    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model checkpoint not found.")
        return
    chk = torch.load(MODEL_FILE, map_location=device, weights_only=False)
    if "model_state_dict" in chk and "user2idx" in chk and "movie2idx" in chk:
        user2idx.update(chk["user2idx"])
        movie2idx.update(chk["movie2idx"])
        model = RecommenderNet(len(user2idx), len(movie2idx), len(content_cols)).to(device)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        print(f"✅ Loaded checkpoint from {MODEL_FILE}")
    else:
        raise RuntimeError("Checkpoint missing required keys.")

def predict_batch(user_idx_int, movie_id_batch):
    u_idx = user_idx_int
    m_idxs = [movie2idx.get(int(mid), 0) for mid in movie_id_batch]
    u = torch.tensor([u_idx]*len(m_idxs), dtype=torch.long, device=device)
    m = torch.tensor(m_idxs, dtype=torch.long, device=device)
    feat_rows = []
    for mid in movie_id_batch:
        if int(mid) in movieid2row:
            feat_rows.append(movies_features.iloc[movieid2row[int(mid)]].values)
        else:
            feat_rows.append(np.zeros((len(content_cols),), dtype=np.float16))
    f = torch.tensor(np.stack(feat_rows).astype(np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(u, m, f)
    return preds.cpu().numpy()

def recommend_for_user(user_id, top_n=10, candidate_limit=500, batch_size=64):
    u_idx = user2idx.get(user_id, 0)
    seen = set()
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        seen = set(df[df["user_id"]==int(user_id)]["movie_id"].astype(int))
    candidates = [mid for mid in movie_lookup.keys() if mid not in seen]
    if len(candidates) > candidate_limit:
        candidates = list(np.random.choice(candidates, candidate_limit, replace=False))
    preds = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        scores = predict_batch(u_idx, batch)
        preds.extend(zip(batch, scores))
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": int(mid), "title": movie_lookup.get(int(mid), "Unknown")} for mid, _ in top_preds]

# -----------------------
# Pydantic models
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
# User helpers
# -----------------------
def get_or_create_user(username: str) -> int:
    username = username.strip().lower()
    if os.path.exists(USERS_FILE):
        users_df = pd.read_csv(USERS_FILE)
    else:
        users_df = pd.DataFrame(columns=["user_id", "username"])
    if username in users_df["username"].values:
        return int(users_df.loc[users_df["username"]==username,"user_id"].iloc[0])
    new_id = int(users_df["user_id"].max()+1) if not users_df.empty else 1
    users_df = pd.concat([users_df, pd.DataFrame([{"user_id": new_id, "username": username}])], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    return new_id

def background_retrain():
    # placeholder for lightweight retrain logic
    print("🔄 Background retrain triggered (not implemented).")

# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
def startup_event():
    try: _load_movies_and_features()
    except Exception as e: print("Failed to load movies/features:", e)
    try: load_model_checkpoint()
    except Exception as e: print("Failed to load model checkpoint:", e)

# -----------------------
# API routes
# -----------------------
@app.post("/recommend/{rec_type}")
def recommend(rec_type: str, req: RecommendRequest):
    user_id = get_or_create_user(req.username)
    recs = recommend_for_user(user_id, top_n=req.top_n)
    return {"recommendations": recs}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id","movie_id","interaction"])
    new_row = {"user_id": int(user_id), "movie_id": int(req.movie_id), "interaction": req.interaction}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(INTERACTIONS_FILE, index=False)
    if len(df) % UPDATE_THRESHOLD == 0:
        threading.Thread(target=background_retrain, daemon=True).start()
    return {"status":"success","message":"Feedback saved."}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    df = pd.read_csv(INTERACTIONS_FILE) if os.path.exists(INTERACTIONS_FILE) else pd.DataFrame(columns=["user_id","movie_id","interaction"])
    user_rows = df[df["user_id"]==user_id]
    return {"history":[{"movieId": int(mid), "title": movie_lookup.get(int(mid),"Unknown"), "interaction": inter} for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"])]}

@app.get("/warmup")
def warmup():
    if model is None:
        try: load_model_checkpoint()
        except Exception as e: return {"status":"error","detail":str(e)}
    return {"status":"ready","device": str(device)}
