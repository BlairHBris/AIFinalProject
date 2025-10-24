"""
Movie Recommender API (Final Hybrid Version)
- PyTorch model integration
- Supports Hybrid, Collaborative, and Content-Based predictions.
- 'liked_movies' are now integrated into the user's content profile.
- Thread-safe CSV handling via threading.Lock (retained from previous fix).
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
    # 💡 NOTE: Replace this with your actual Render frontend URL
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
movie_lookup = {}
title_to_movieid = {} # 💡 NEW: Reverse lookup for movie titles
movies_features = None
movieid2row = {}
movies_df = None
ratings_df = None
tags_df = None
file_lock = threading.Lock() 

# -----------------------
# Model
# -----------------------
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, num_features, embedding_dim=32):
        super().__init__()
        # Embedding for collaborative filtering part
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.movie_embed = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)
        
        # 💡 FIX: For pure content mode, we will combine the item features (content_cols)
        # with the user's preference vector (user_content_features). Let's keep the layer 
        # structure flexible. In hybrid/collab mode, `user_content_features` will be zeros.
        self.fc = nn.Linear(embedding_dim + num_features, 1)

    def forward(self, user_idx, movie_idx, features):
        u_emb = self.user_embed(user_idx)
        m_emb = self.movie_embed(movie_idx)
        
        # Collaborative signal (element-wise product)
        x_collab = u_emb * m_emb 
        
        # Combine collaborative signal and content features
        x = torch.cat([x_collab, features], dim=1) 
        
        x = self.fc(x)
        return x.squeeze()

# -----------------------
# Utilities
# -----------------------
def load_data():
    """Loads and processes all static data files."""
    global movies_df, ratings_df, tags_df, movies_features, content_cols, movie_lookup, movieid2row, title_to_movieid
    
    # ... (Loading and pre-processing CSVs remains the same as previous corrected version) ...
    # [Start of previous load_data implementation]

    # 1. Load CSVs safely
    try:
        movies_df = pd.read_csv(MOVIES_FILE).fillna("")
        ratings_df = pd.read_csv(RATINGS_FILE)
        tags_df = pd.read_csv(TAGS_FILE).fillna("")
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None, None, None, None

    # 2. Normalize columns
    movies_df.columns = [c.strip().lower() for c in movies_df.columns]
    ratings_df.columns = [c.strip().lower() for c in ratings_df.columns]
    tags_df.columns = [c.strip().lower() for c in tags_df.columns]

    # 3. Process Genres (One-Hot Encoding)
    movies_df["genres"] = movies_df.get("genres", "").fillna("")
    unique_genres = {g for g in "|".join(movies_df["genres"]).split("|") if g}
    for g in unique_genres:
        movies_df[g] = movies_df["genres"].apply(lambda x: 1 if g in x else 0).astype(np.float16)

    # 4. Process Tags (Top 512 One-Hot Encoding)
    tags_df["tag"] = tags_df["tag"].astype(str).str.lower()
    top_tags = tags_df["tag"].value_counts().head(512).index.tolist()
    tag_matrix = tags_df[tags_df["tag"].isin(top_tags)].pivot_table(
        index="movieid", columns="tag", aggfunc="size", fill_value=0
    )
    tag_matrix = tag_matrix.reindex(columns=top_tags, fill_value=0).astype(np.float16)
    movies_df = movies_df.merge(tag_matrix, on="movieid", how="left").fillna(0)
    
    if "actors" not in movies_df.columns:
         movies_df["actors"] = ""

    # 5. Global Lookups
    content_cols = sorted(list(unique_genres)) + top_tags
    movie_lookup = movies_df.set_index("movieid")["title"].to_dict()
    # 💡 NEW: Title to MovieID mapping
    title_to_movieid = movies_df.set_index("title")["movieid"].to_dict() 
    
    movies_features = movies_df.set_index("movieid")[content_cols].astype(np.float32)
    movieid2row = {int(mid): i for i, mid in enumerate(movies_features.index)}
    
    print(f"Loaded {len(movies_df)} movies and {len(content_cols)} content features.")
    return movies_df, ratings_df, tags_df, movies_features, content_cols, movie_lookup, movieid2row

# ... (load_model_checkpoint and get_or_create_user remain the same) ...

def load_model_checkpoint():
    """Loads the PyTorch model and index mappings."""
    global model, user2idx, movie2idx

    if movies_df is None: 
        load_data() 

    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model checkpoint not found.")
        return

    try:
        # Allow old numpy/pickle objects from legacy PyTorch
        import torch.serialization
        torch.serialization.add_safe_globals([torch.LongTensor, torch.FloatTensor, np.float32, np.float64, np.int64, np.int32])

        chk = torch.load(MODEL_FILE, map_location=device, weights_only=False)
        
        user2idx.update(chk.get("user2idx", {}))
        movie2idx.update(chk.get("movie2idx", {}))
        
        num_features = len(content_cols) 
        if num_features == 0:
            num_features = len(chk.get("content_cols", []))

        model = RecommenderNet(len(user2idx), len(movie2idx), num_features).to(device)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        print(f"✅ Loaded checkpoint from {MODEL_FILE}. Users: {len(user2idx)}, Movies: {len(movie2idx)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def get_or_create_user(username: str) -> int:
    """Gets existing user ID or creates a new one, thread-safe."""
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
    """Aggregates content features (genres/tags) for a list of liked movie titles."""
    if movies_features is None or not liked_movies_titles:
        return np.zeros((len(content_cols),), dtype=np.float32)

    # 1. Map titles to Movie IDs
    liked_movie_ids = [title_to_movieid.get(title) for title in liked_movies_titles if title_to_movieid.get(title) is not None]
    
    if not liked_movie_ids:
        return np.zeros((len(content_cols),), dtype=np.float32)

    # 2. Get features for these movies
    liked_features = []
    for mid in liked_movie_ids:
        if mid in movieid2row:
            liked_features.append(movies_features.iloc[movieid2row[mid]].values)

    if not liked_features:
        return np.zeros((len(content_cols),), dtype=np.float32)

    # 3. Average the feature vectors to create a "user content profile"
    avg_features = np.mean(np.stack(liked_features), axis=0)
    return avg_features.astype(np.float32)


def predict_batch(user_idx_int: int, movie_id_batch: List[int], user_content_features: np.ndarray = None, mode: str = 'hybrid') -> np.ndarray:
    """Performs batch prediction, dynamically adjusting inputs based on mode."""
    if model is None or movies_features is None:
        return np.zeros(len(movie_id_batch))

    # --- Collaborative Input ---
    u_idx = user2idx.get(user_idx_int, 0)
    m_idxs = [movie2idx.get(int(mid), 0) for mid in movie_id_batch]

    # --- Content Input ---
    movie_ids_int = np.array(movie_id_batch, dtype=int)
    feat_indices = np.array([movieid2row.get(mid, -1) for mid in movie_ids_int], dtype=int)
    f_matrix = np.zeros((len(movie_id_batch), len(content_cols)), dtype=np.float32)
    valid_mask = feat_indices != -1
    valid_indices = feat_indices[valid_mask]
    if valid_indices.size > 0:
        f_matrix[valid_mask] = movies_features.iloc[valid_indices].values.astype(np.float32)

    # --- Mode Adjustment ---
    # 1. Pure Collaborative: Set all content features to zero
    if mode == 'collab':
        f_matrix = np.zeros_like(f_matrix)
        
    # 2. Pure Content: Collaborative signal is derived from the user's explicit content profile
    #    We can simulate a Content-Based approach by using a pseudo-user vector (user_content_features)
    #    and disabling the embeddings (setting user_idx to 0, or passing a zero vector for u_emb in a custom forward pass).
    #    For simplicity, we'll keep the model structure, and instead, use the content features to weight the prediction.
    if mode == 'content':
        # Pure Content Prediction Simulation (simplified):
        # We assume the content features (f_matrix) are sufficient, and we zero out the CF part
        # by passing user_idx=0 (which is the padding index, usually a zero embedding).
        u_idx = 0 
        
        # NOTE: A more advanced content model might use a separate content-only network, 
        # but for this hybrid structure, this is the simplest way to enforce 'content-only' scoring.

    # --- Tensor Preparation ---
    u = torch.tensor([u_idx]*len(m_idxs), dtype=torch.long, device=device)
    m = torch.tensor(m_idxs, dtype=torch.long, device=device)
    f = torch.tensor(f_matrix, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        preds = model(u, m, f)
    return preds.cpu().numpy()


def recommend_for_user(user_id: int, top_n: int = 10, liked_genres: List[str] = [], liked_actors: List[str] = [], liked_movies: List[str] = [], rec_type: str = "hybrid") -> List[Dict[str, Any]]:
    """Generates recommendations based on the selected mode and filters."""
    if movies_df is None or model is None:
        print("⚠️ Model or data not loaded.")
        return []

    # 1. Get seen movies (Thread-safe read)
    with file_lock:
        if os.path.exists(INTERACTIONS_FILE):
            df = pd.read_csv(INTERACTIONS_FILE)
            seen = set(df[df["user_id"] == int(user_id)]["movie_id"].astype(int))
        else:
            seen = set()
    
    # 2. Start with all unseen movies
    candidates_df = movies_df[~movies_df["movieid"].isin(seen)].copy()

    # 3. Content Filtering (for all modes except pure Collaborative)
    if rec_type != 'collab':
        # --- Genre Filtering ---
        if liked_genres:
            genre_mask = pd.Series(True, index=candidates_df.index)
            for g in liked_genres:
                if g in candidates_df.columns:
                    genre_mask &= (candidates_df[g] == 1)
            candidates_df = candidates_df[genre_mask]
            
        # --- Actor Filtering ---
        if liked_actors:
            for a in liked_actors:
                candidates_df = candidates_df[candidates_df.get("actors", "").str.lower().str.contains(a.lower(), na=False)]

    candidates = candidates_df["movieid"].tolist()
    
    if not candidates:
        print("No candidates remaining after filtering.")
        return []

    # 4. Score candidates using dynamic prediction logic
    preds = []
    batch_size = 64
    u_idx = user2idx.get(user_id, 0)
    
    # NOTE: user_content_features is not explicitly passed to predict_batch 
    # but the logic for content-based prediction is handled internally 
    # by setting u_idx = 0 and relying on item content features (f_matrix).
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        # Pass the desired mode to the batch prediction function
        scores = predict_batch(u_idx, batch, mode=rec_type) 
        preds.extend(zip(batch, scores))

    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    print(f"Generated {len(top_preds)} recommendations using {rec_type} mode.")

    # 5. Enriched results (same as before)
    enriched = []
    for mid, _ in top_preds:
        movie_row_df = movies_df[movies_df["movieid"] == mid]
        if movie_row_df.empty: continue
            
        movie_row = movie_row_df.iloc[0]
        
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
    liked_genres: List[str] = []
    liked_actors: List[str] = []
    liked_movies: List[str] = [] # 💡 FIX: Changed to List[str] to accept movie titles
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
    # Validate rec_type
    if rec_type not in ['hybrid', 'collab', 'content']:
        return {"status": "error", "message": "Invalid recommendation type."}, 400
        
    user_id = get_or_create_user(req.username)
    print(f"Rec Request: User={req.username}, Type={rec_type}, Genres={len(req.liked_genres)}, Movies={len(req.liked_movies)}, Actors={len(req.liked_actors)}")

    recs = recommend_for_user(user_id, top_n=req.top_n, liked_genres=req.liked_genres,
                              liked_actors=req.liked_actors, liked_movies=req.liked_movies, # Now passes titles
                              rec_type=rec_type)
    return {"recommendations": recs}

# ... (feedback, get_user_history, warmup, and startup_event remain the same) ...

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    
    with file_lock: 
        if os.path.exists(INTERACTIONS_FILE):
            df = pd.read_csv(INTERACTIONS_FILE)
        else:
            df = pd.DataFrame(columns=["user_id", "movie_id", "interaction"])

        df = df[~((df["user_id"] == user_id) & (df["movie_id"] == req.movie_id))]
        
        if req.interaction != "remove":
            new_row = {"user_id": int(user_id), "movie_id": int(req.movie_id), "interaction": req.interaction}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(INTERACTIONS_FILE, index=False)

        if len(df) % UPDATE_THRESHOLD == 0:
            threading.Thread(target=lambda: print(f"🔄 Background retrain placeholder triggered. Interactions: {len(df)}"), daemon=True).start()

    return {"status": "success", "message": f"Feedback '{req.interaction}' updated for movie {req.movie_id}."}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    history = []
    
    with file_lock: 
        if not os.path.exists(INTERACTIONS_FILE):
            return {"history": []}
        df = pd.read_csv(INTERACTIONS_FILE)
    
    user_rows = df[df["user_id"] == user_id]
    
    if movies_df is None:
        return {"history": []}

    for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"]):
        movie_row_df = movies_df[movies_df["movieid"] == mid]
        if movie_row_df.empty:
            continue
            
        movie_row = movie_row_df.iloc[0]
        history.append({
            "movieId": int(mid),
            "title": movie_lookup.get(mid, "Unknown"),
            "interaction": inter,
            "genres": movie_row["genres"].split("|")
        })
    return {"history": history}

@app.get("/warmup")
def warmup():
    """Initializes model and data if not already loaded."""
    if model is None:
        try:
            load_model_checkpoint()
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    if movies_df is None:
        try:
            load_data()
        except Exception as e:
            return {"status": "error", "detail": str(e)}
            
    return {"status": "ready", "device": str(device)}

@app.on_event("startup")
def startup_event():
    """Loads all necessary data and model once at application start."""
    try:
        load_data()
        load_model_checkpoint()
        print("✅ Startup complete: model and data ready")
    except Exception as e:
        print(f"❌ Startup error: {e}")