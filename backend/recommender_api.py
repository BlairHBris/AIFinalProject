"""
Movie Recommender API (Optimized)
- PyTorch model integration
- Hybrid recommendation (content + collaborative)
- Enriched info: avg rating, genres, top tags
- Safe CORS setup for multiple origins
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
# Globals
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
user2idx = {}
movie2idx = {}
content_cols = []
movie_lookup = {}
movies_features = None
movieid2row = {}
movies_df = None
ratings_df = None
tags_df = None

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
def load_data():
    global movies_df, ratings_df, tags_df, movies_features, content_cols, movie_lookup, movieid2row
    # Load CSVs safely
    movies_df = pd.read_csv(MOVIES_FILE).fillna("")
    ratings_df = pd.read_csv(RATINGS_FILE)
    tags_df = pd.read_csv(TAGS_FILE).fillna("")

    # Normalize columns
    movies_df.columns = [c.strip().lower() for c in movies_df.columns]
    ratings_df.columns = [c.strip().lower() for c in ratings_df.columns]
    tags_df.columns = [c.strip().lower() for c in tags_df.columns]

    if "actors" not in movies_df.columns:
        movies_df["actors"] = ""

    movies_df["genres"] = movies_df.get("genres", "").fillna("")
    unique_genres = {g for g in "|".join(movies_df["genres"]).split("|") if g}
    for g in unique_genres:
        movies_df[g] = movies_df["genres"].apply(lambda x: 1 if g in x else 0).astype(np.float16)

    # Process tags (top 512)
    tags_df["tag"] = tags_df["tag"].astype(str).str.lower()
    top_tags = tags_df["tag"].value_counts().head(512).index.tolist()
    tag_matrix = tags_df[tags_df["tag"].isin(top_tags)].pivot_table(
        index="movieid", columns="tag", aggfunc="size", fill_value=0
    )
    tag_matrix = tag_matrix.reindex(columns=top_tags, fill_value=0).astype(np.float16)
    movies_df = movies_df.merge(tag_matrix, on="movieid", how="left").fillna(0)

    content_cols = sorted(list(unique_genres)) + top_tags
    movie_lookup = movies_df.set_index("movieid")["title"].to_dict()
    movies_features = movies_df.set_index("movieid")[content_cols].astype(np.float16)
    movieid2row = {int(mid): i for i, mid in enumerate(movies_features.index)}

    return movies_df, ratings_df, tags_df, movies_features, content_cols, movie_lookup, movieid2row

def load_model_checkpoint():
    global model, user2idx, movie2idx
    _, _, _, _, _, _, _ = load_data()
    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model checkpoint not found.")
        return

    chk = torch.load(MODEL_FILE, map_location=device)
    user2idx.update(chk.get("user2idx", {}))
    movie2idx.update(chk.get("movie2idx", {}))
    model = RecommenderNet(len(user2idx), len(movie2idx), len(chk.get("content_cols", []))).to(device)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint from {MODEL_FILE}")

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

def recommend_for_user(user_id, top_n=10, liked_genres=[], liked_actors=[], liked_movies=[], rec_type="hybrid"):
    seen = set()
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        seen = set(df[df["user_id"] == int(user_id)]["movie_id"].astype(int))
    candidates = [mid for mid in movie_lookup.keys() if mid not in seen]

    # Filter candidates
    filtered = []
    for mid in candidates:
        row = movies_df[movies_df["movieid"] == mid].iloc[0]
        genre_ok = all(row.get(g, 0) == 1 for g in liked_genres)
        actor_ok = all(a.lower() in row.get("actors", "").lower() for a in liked_actors)
        movie_ok = mid in liked_movies if liked_movies else True
        if genre_ok and actor_ok and movie_ok:
            filtered.append(mid)
    candidates = filtered if filtered else candidates

    # Score
    preds = []
    batch_size = 64
    u_idx = user2idx.get(user_id, 0)
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        scores = predict_batch(u_idx, batch)
        preds.extend(zip(batch, scores))

    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

    enriched = []
    for mid, _ in top_preds:
        movie_row = movies_df[movies_df["movieid"] == mid].iloc[0]
        movie_ratings = ratings_df[ratings_df["movieid"] == mid]["rating"]
        avg_rating = movie_ratings.mean() if not movie_ratings.empty else None
        tags = tags_df[tags_df["movieid"] == mid]["tag"].value_counts().head(3).index.tolist()
        enriched.append({
            "movieId": int(mid),
            "title": movie_lookup.get(mid, "Unknown"),
            "avg_rating": avg_rating,
            "genres": movie_row["genres"].split("|"),
            "top_tags": tags
        })
    return enriched

# -----------------------
# Pydantic Models
# -----------------------
class RecommendRequest(BaseModel):
    username: str
    liked_genres: list[str] = []
    liked_actors: list[str] = []
    liked_movies: list[int] = []
    top_n: int = 5

class FeedbackRequest(BaseModel):
    username: str
    movie_id: int
    interaction: str

# -----------------------
# Routes
# -----------------------
@app.post("/recommend/{rec_type}")
def recommend(rec_type: str, req: RecommendRequest):
    user_id = get_or_create_user(req.username)
    recs = recommend_for_user(user_id, top_n=req.top_n, liked_genres=req.liked_genres,
                              liked_actors=req.liked_actors, liked_movies=req.liked_movies,
                              rec_type=rec_type)
    return {"recommendations": recs}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
    else:
        df = pd.DataFrame(columns=["user_id", "movie_id", "interaction"])

    # Handle removal
    if req.interaction == "remove":
        df = df[~((df["user_id"] == user_id) & (df["movie_id"] == req.movie_id))]
    else:
        # Remove existing same-type feedback first (optional: ensures one per type)
        df = df[~((df["user_id"] == user_id) & (df["movie_id"] == req.movie_id) & (df["interaction"] == req.interaction))]
        new_row = {"user_id": int(user_id), "movie_id": int(req.movie_id), "interaction": req.interaction}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(INTERACTIONS_FILE, index=False)

    # Background retrain trigger
    if len(df) % UPDATE_THRESHOLD == 0:
        threading.Thread(target=lambda: print("🔄 Background retrain placeholder"), daemon=True).start()

    return {"status": "success", "message": "Feedback updated."}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    if not os.path.exists(INTERACTIONS_FILE):
        return {"history": []}
    df = pd.read_csv(INTERACTIONS_FILE)
    user_rows = df[df["user_id"] == user_id]
    history = []
    for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"]):
        movie_row = movies_df[movies_df["movieid"] == mid].iloc[0]
        history.append({
            "movieId": int(mid),
            "title": movie_lookup.get(mid, "Unknown"),
            "interaction": inter,
            "genres": movie_row["genres"].split("|")
        })
    return {"history": history}

@app.get("/warmup")
def warmup():
    if model is None:
        try:
            load_model_checkpoint()
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    return {"status": "ready", "device": str(device)}

# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
def startup_event():
    try:
        load_data()
        load_model_checkpoint()
        print("✅ Startup complete: model and data ready")
    except Exception as e:
        print(f"❌ Startup error: {e}")
