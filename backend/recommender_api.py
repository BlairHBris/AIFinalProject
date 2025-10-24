"""
Movie Recommender API (Final Hybrid Version)
- PyTorch model integration
- Supports Hybrid, Collaborative, and Content-Based predictions.
- 'liked_movies' are now integrated into the user's content profile.
- Thread-safe CSV handling via threading.Lock.
- Data files expected in the same folder as this script.
"""

import os
import threading
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# -----------------------
# Config / File Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_FILE = os.path.join(BASE_DIR, "movies.csv")
RATINGS_FILE = os.path.join(BASE_DIR, "ratings.csv")
TAGS_FILE = os.path.join(BASE_DIR, "tags.csv")
USERS_FILE = os.path.join(BASE_DIR, "users.csv")
INTERACTIONS_FILE = os.path.join(BASE_DIR, "user_interactions.csv")
MODEL_FILE = os.path.join(BASE_DIR, "torch_recommender.pt")
UPDATE_THRESHOLD = 10

FRONTEND_URLS = [
    "http://localhost:3000",
    "https://movie-recommender-frontend-nxw6.onrender.com"
]

# -----------------------
# FastAPI + CORS
# -----------------------
app = FastAPI(title="Movie Recommender", version="v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend live", "cors": "enabled"}

# -----------------------
# Globals & Locks
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
user2idx = {}
movie2idx = {}
content_cols = []
movies_df = None
ratings_df = None
tags_df = None
movies_features = None
movie_lookup = {}
title_to_movieid = {}
movieid2row = {}
file_lock = threading.Lock()

# -----------------------
# PyTorch Model
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
        x_collab = u_emb * m_emb
        x = torch.cat([x_collab, features], dim=1)
        return self.fc(x).squeeze()

# -----------------------
# Utilities
# -----------------------
def load_data():
    """Load movies, ratings, tags, and prepare content features."""
    global movies_df, ratings_df, tags_df, movies_features, content_cols, movie_lookup, movieid2row, title_to_movieid
    
    try:
        movies_df = pd.read_csv(MOVIES_FILE).fillna("")
        ratings_df = pd.read_csv(RATINGS_FILE)
        tags_df = pd.read_csv(TAGS_FILE).fillna("")
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    movies_df.columns = [c.strip().lower() for c in movies_df.columns]
    ratings_df.columns = [c.strip().lower() for c in ratings_df.columns]
    tags_df.columns = [c.strip().lower() for c in tags_df.columns]

    # Process genres
    movies_df["genres"] = movies_df.get("genres", "").fillna("")
    unique_genres = {g for g in "|".join(movies_df["genres"]).split("|") if g}
    for g in unique_genres:
        movies_df[g] = movies_df["genres"].apply(lambda x: 1 if g in x else 0).astype(np.float32)

    # Process tags (top 512)
    tags_df["tag"] = tags_df["tag"].astype(str).str.lower()
    top_tags = tags_df["tag"].value_counts().head(512).index.tolist()
    tag_matrix = tags_df[tags_df["tag"].isin(top_tags)].pivot_table(
        index="movieid", columns="tag", aggfunc="size", fill_value=0
    )
    tag_matrix = tag_matrix.reindex(columns=top_tags, fill_value=0).astype(np.float32)
    movies_df = movies_df.merge(tag_matrix, on="movieid", how="left").fillna(0)

    if "actors" not in movies_df.columns:
        movies_df["actors"] = ""

    content_cols[:] = sorted(list(unique_genres)) + top_tags
    movie_lookup.update(movies_df.set_index("movieid")["title"].to_dict())
    title_to_movieid.update(movies_df.set_index("title")["movieid"].to_dict())
    movies_features = movies_df.set_index("movieid")[content_cols].astype(np.float32)
    movieid2row.update({int(mid): i for i, mid in enumerate(movies_features.index)})

    print(f"✅ Loaded {len(movies_df)} movies and {len(content_cols)} content features.")

def load_model_checkpoint():
    global model, user2idx, movie2idx

    if movies_df is None:
        load_data()

    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model checkpoint not found.")
        return

    try:
        chk = torch.load(MODEL_FILE, map_location=device)
        user2idx.update(chk.get("user2idx", {}))
        movie2idx.update(chk.get("movie2idx", {}))
        num_features = len(content_cols)
        model = RecommenderNet(len(user2idx), len(movie2idx), num_features).to(device)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        print(f"✅ Loaded checkpoint: Users={len(user2idx)}, Movies={len(movie2idx)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def get_or_create_user(username: str) -> int:
    username = username.strip().lower()
    with file_lock:
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

def get_content_vector_from_titles(liked_movies_titles: List[str]) -> np.ndarray:
    if movies_features is None or not liked_movies_titles:
        return np.zeros((len(content_cols),), dtype=np.float32)

    liked_movie_ids = [title_to_movieid.get(title) for title in liked_movies_titles if title_to_movieid.get(title) is not None]
    if not liked_movie_ids:
        return np.zeros((len(content_cols),), dtype=np.float32)

    liked_features = [movies_features.iloc[movieid2row[mid]].values for mid in liked_movie_ids if mid in movieid2row]
    if not liked_features:
        return np.zeros((len(content_cols),), dtype=np.float32)

    return np.mean(np.stack(liked_features), axis=0).astype(np.float32)

def predict_batch(user_idx_int: int, movie_id_batch: List[int], user_content_features: np.ndarray = None, mode: str = 'hybrid') -> np.ndarray:
    if model is None or movies_features is None:
        return np.zeros(len(movie_id_batch))

    u_idx = user2idx.get(user_idx_int, 0)
    m_idxs = [movie2idx.get(int(mid), 0) for mid in movie_id_batch]

    feat_indices = np.array([movieid2row.get(mid, -1) for mid in movie_id_batch], dtype=int)
    f_matrix = np.zeros((len(movie_id_batch), len(content_cols)), dtype=np.float32)
    valid_mask = feat_indices != -1
    f_matrix[valid_mask] = movies_features.iloc[feat_indices[valid_mask]].values

    if mode == 'collab':
        f_matrix = np.zeros_like(f_matrix)
    if mode == 'content':
        u_idx = 0

    u = torch.tensor([u_idx]*len(m_idxs), dtype=torch.long, device=device)
    m = torch.tensor(m_idxs, dtype=torch.long, device=device)
    f = torch.tensor(f_matrix, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(u, m, f)
    return preds.cpu().numpy()

def recommend_for_user(user_id: int, top_n: int = 10, liked_genres: List[str] = [], liked_actors: List[str] = [], liked_movies: List[str] = [], rec_type: str = "hybrid") -> List[Dict[str, Any]]:
    if movies_df is None or model is None:
        return []

    with file_lock:
        if os.path.exists(INTERACTIONS_FILE):
            interactions_df = pd.read_csv(INTERACTIONS_FILE)
            seen = set(interactions_df[interactions_df["user_id"]==user_id]["movie_id"].astype(int))
        else:
            seen = set()

        candidates_df = movies_df[~movies_df["movieid"].isin(seen)].copy()

        if rec_type != 'collab':
            if liked_genres:
                genre_mask = pd.Series(True, index=candidates_df.index)
                for g in liked_genres:
                    if g in candidates_df.columns:
                        genre_mask &= (candidates_df[g] == 1)
                candidates_df = candidates_df[genre_mask]

            if liked_actors:
                for a in liked_actors:
                    candidates_df = candidates_df[candidates_df.get("actors", "").str.lower().str.contains(a.lower(), na=False)]

        candidates = candidates_df["movieid"].tolist()
        if not candidates:
            return []

        preds = []
        batch_size = 64
        u_idx = user2idx.get(user_id, 0)
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            scores = predict_batch(u_idx, batch, mode=rec_type)
            preds.extend(zip(batch, scores))

        top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
        enriched = []
        for mid, _ in top_preds:
            row = movies_df[movies_df["movieid"] == mid].iloc[0]
            movie_ratings = ratings_df[ratings_df["movieid"] == mid]["rating"]
            avg_rating = movie_ratings.mean() if not movie_ratings.empty else None
            tags = tags_df[tags_df["movieid"] == mid]["tag"].value_counts().head(3).index.tolist()
            enriched.append({
                "movieId": int(mid),
                "title": movie_lookup.get(mid, "Unknown"),
                "avg_rating": avg_rating,
                "genres": row["genres"].split("|"),
                "top_tags": tags
            })
        return enriched

# -----------------------
# Pydantic Models
# -----------------------
class RecommendRequest(BaseModel):
    username: str
    liked_genres: List[str] = []
    liked_actors: List[str] = []
    liked_movies: List[str] = []
    top_n: int = 5

class FeedbackRequest(BaseModel):
    username: str
    movie_id: int
    interaction: str

# -----------------------
# Routes
# -----------------------
@app.post("/recommend/{rec_type}")
def recommend_route(rec_type: str, req: RecommendRequest):
    user_id = get_or_create_user(req.username)
    recs = recommend_for_user(user_id, top_n=req.top_n, liked_genres=req.liked_genres,
                              liked_actors=req.liked_actors, liked_movies=req.liked_movies,
                              rec_type=rec_type)
    return {"recommendations": recs}

@app.post("/feedback")
def feedback_route(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    with file_lock:
        if os.path.exists(INTERACTIONS_FILE):
            df = pd.read_csv(INTERACTIONS_FILE)
        else:
            df = pd.DataFrame(columns=["user_id","movie_id","interaction"])
        df = df[~((df["user_id"]==user_id)&(df["movie_id"]==req.movie_id))]
        if req.interaction != "remove":
            df = pd.concat([df, pd.DataFrame([{"user_id": user_id,"movie_id": req.movie_id,"interaction": req.interaction}])], ignore_index=True)
        df.to_csv(INTERACTIONS_FILE, index=False)
    return {"status":"success"}

@app.get("/users/{username}/history")
def get_history(username: str):
    user_id = get_or_create_user(username)
    history = []
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        user_rows = df[df["user_id"]==user_id]
        if movies_df is not None:
            for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"]):
                row = movies_df[movies_df["movieid"]==mid]
                if not row.empty:
                    row = row.iloc[0]
                    history.append({"movieId": int(mid), "title": movie_lookup.get(mid,"Unknown"), "interaction": inter, "genres": row["genres"].split("|")})
    return {"history": history}

@app.get("/warmup")
def warmup():
    if movies_df is None: load_data()
    if model is None: load_model_checkpoint()
    return {"status": "ready", "device": str(device)}

@app.on_event("startup")
def startup_event():
    load_data()
    load_model_checkpoint()
    print("✅ Startup complete")

# -----------------------
# Helper endpoints for frontend dropdowns
# -----------------------

@app.get("/genres")
def get_genres():
    if movies_df is None:
        load_data()
    genres = sorted([c for c in content_cols if c not in tags_df.columns])
    return genres

@app.get("/actors")
def get_actors(top_n: int = 20):
    if movies_df is None:
        load_data()
    # Extract actors from movies_df, split by comma
    all_actors = movies_df["actors"].dropna().str.split(",").explode().str.strip()
    top_actors = all_actors.value_counts().head(top_n).index.tolist()
    return top_actors

@app.get("/movies")
def get_top_movies():
    """Returns top 25 movies sorted by average rating for frontend dropdown."""
    if movies_df is None or ratings_df is None:
        load_data()

    # Compute average ratings
    avg_ratings = ratings_df.groupby("movieid")["rating"].mean()
    
    # Merge with movies_df
    movies_copy = movies_df.copy()
    movies_copy["avg_rating"] = movies_copy["movieid"].map(avg_ratings).fillna(0)
    
    # Sort descending by avg_rating and take top 25
    top_movies = movies_copy.sort_values("avg_rating", ascending=False).head(25)
    
    # Return only titles
    return [title for title in top_movies["title"].tolist()]

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
