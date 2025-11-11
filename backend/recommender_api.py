import os
import threading
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
import re
from fastapi import FastAPI, HTTPException
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
TOP_GENRE_COUNT = 20 
TOP_MOVIE_COUNT = 20 
TMDB_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzNTg2YTg2MWI2MmM3ZDczNzM5MWM0MTgyNzhmODIwNSIsIm5iZiI6MTc2MjU0OTIwOS44NTQsInN1YiI6IjY5MGU1ZGQ5YTc3MGZmMzhjNWMwNTMxZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.liYjcu3qm4JdIGhVjmchD2zebG-JCQQvjarTSVVRDd8"
TMDB_BASE_URL = "https://api.themoviedb.org/3/search/movie?"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w200"

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

    # --- Process Genres (Standardization and Reduction) ---
    movies_df["genres"] = movies_df.get("genres", "").fillna("")
    
    # 1. Standardize to Title Case and count frequency
    all_genres = movies_df["genres"].str.split("|").explode().str.title().dropna()
    genre_counts = all_genres.value_counts()
    
    # 2. Select top N genres
    top_genres = genre_counts.head(TOP_GENRE_COUNT).index.tolist()
    unique_genres = set(top_genres)
    
    # 3. One-hot encode only the top genres
    for g in top_genres:
        movies_df[g] = movies_df["genres"].apply(
            lambda x: 1 if g in x.title() else 0
        ).astype(np.float32)

    # Process tags (top 512)
    tags_df["tag"] = tags_df["tag"].astype(str).str.lower()
    top_tags = tags_df["tag"].value_counts().head(512).index.tolist()
    tag_matrix = tags_df[tags_df["tag"].isin(top_tags)].pivot_table(
        index="movieid", columns="tag", aggfunc="size", fill_value=0
    )
    tag_matrix = tag_matrix.reindex(columns=top_tags, fill_value=0).astype(np.float32)
    movies_df = movies_df.merge(tag_matrix, on="movieid", how="left").fillna(0)

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
        # FIX: Set weights_only=False for reliable PyTorch loading
        chk = torch.load(MODEL_FILE, map_location=device, weights_only=False) 
        
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
    
def get_history_content_vector(user_id: int) -> np.ndarray:
    """Calculates an average feature vector from a user's 'interested' or 'watched' history."""
    if movies_features is None:
        return np.zeros((len(content_cols),), dtype=np.float32)
        
    with file_lock:
        interactions_df = pd.DataFrame(columns=["user_id", "movie_id", "interaction"]) 
        if os.path.exists(INTERACTIONS_FILE) and os.path.getsize(INTERACTIONS_FILE) > 0:
            try:
                interactions_df = pd.read_csv(INTERACTIONS_FILE)
            except pd.errors.EmptyDataError:
                pass 
    
    # Filter for 'interested' or 'watched' interactions (positive signals)
    positive_interactions = interactions_df[
        (interactions_df["user_id"] == user_id) & 
        (interactions_df["interaction"].isin(['interested', 'watched']))
    ]

    if positive_interactions.empty:
        return np.zeros((len(content_cols),), dtype=np.float32)

    # Get the features for these movies
    positive_movie_ids = positive_interactions["movie_id"].astype(int).tolist()
    
    liked_features = [movies_features.iloc[movieid2row[mid]].values 
                    for mid in positive_movie_ids 
                    if mid in movieid2row]

    if not liked_features:
        return np.zeros((len(content_cols),), dtype=np.float32)

    # Average the feature vectors to create a content profile
    return np.mean(np.stack(liked_features), axis=0).astype(np.float32)


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
# --- END NEW/MODIFIED UTILITY FUNCTIONS ---


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


# --- recommend_for_user ---
def recommend_for_user(user_id: int, top_n: int = 12, liked_genres: List[str] = [], liked_movies: List[str] = [], rec_type: str = "hybrid") -> List[Dict[str, Any]]:
    if movies_df is None or model is None:
        return []

    # --- 1. Get Seen Movies & Initialize Candidates (Exclusion Logic) ---
    with file_lock:
        seen = set()
        interactions_df = pd.DataFrame(columns=["user_id", "movie_id", "interaction"]) 
        if os.path.exists(INTERACTIONS_FILE) and os.path.getsize(INTERACTIONS_FILE) > 0:
            try:
                interactions_df = pd.read_csv(INTERACTIONS_FILE)
            except pd.errors.EmptyDataError:
                pass 
        
        # EXCLUSION: Identify all movies user has interacted with
        seen = set(interactions_df[interactions_df["user_id"]==user_id]["movie_id"].astype(int))
        # Candidates are ALL unseen movies
        candidates_df = movies_df[~movies_df["movieid"].isin(seen)].copy()

        candidates = candidates_df["movieid"].tolist()
        if not candidates:
            return []

    # --- 2. Score Candidates (Prediction) ---
    preds = []
    batch_size = 64
    u_idx = user2idx.get(user_id, 0)
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        scores = predict_batch(u_idx, batch, mode=rec_type)
        preds.extend(zip(batch, scores))

    # --- 3. APPLY SOFT BIASING (UPDATED LOGIC: Genre + Similarity Boost) ---
    final_scores = []
    GENRE_BOOST = 1.0 # Base Max boost for history/similarity 
    EXPLICIT_GENRE_BOOST = 5.0 # High boost for explicit user selection

    # NEW: Calculate content profile from Liked Movies and History
    if rec_type != 'collab':
        
        # 3a. Get content vector from explicitly selected movies
        liked_movies_vector = get_content_vector_from_titles(liked_movies)

        # 3b. Get content vector from positive interaction history
        history_vector = get_history_content_vector(user_id)
        
        # 3c. Combine vectors (simple average of explicit and history signals)
        vectors = [v for v in [liked_movies_vector, history_vector] if np.any(v != 0)]
        if vectors:
            user_content_profile = np.mean(np.stack(vectors), axis=0).astype(np.float32)
            # Normalize the combined vector for cosine similarity scaling
            user_content_profile = user_content_profile / (np.linalg.norm(user_content_profile) + 1e-6)
        else:
            user_content_profile = np.zeros((len(content_cols),), dtype=np.float32)
    else:
        user_content_profile = np.zeros((len(content_cols),), dtype=np.float32)


    # 3d. Apply soft bias based on selected Genres AND Content Profile Similarity
    for mid, score in preds:
        movie_row_df = candidates_df[candidates_df["movieid"] == mid]
        if movie_row_df.empty:
            continue
            
        movie_row = movie_row_df.iloc[0]
        
        # --- 3d(i) Genre Selection Boost (Existing Logic) ---
        genre_boost = 0.0
        if liked_genres:
            match_count = 0
            # Use the explicit selected genres for a direct, simple boost
            for g in liked_genres:
                if g in candidates_df.columns and movie_row.get(g, 0) == 1:
                    match_count += 1
            genre_boost = (match_count / len(liked_genres)) * EXPLICIT_GENRE_BOOST
        
        # --- 3d(ii) Content Profile Similarity Boost (NEW LOGIC) ---
        # This uses the combination of liked movies AND positive history
        similarity_boost = 0.0
        if np.any(user_content_profile != 0) and rec_type != 'collab':
            
            # Get the candidate movie's feature vector
            movie_features_vector = movies_features.iloc[movieid2row[mid]].values
            
            # Calculate Cosine Similarity (Dot product of normalized vectors)
            movie_features_vector_norm = movie_features_vector / (np.linalg.norm(movie_features_vector) + 1e-6)
            
            similarity = np.dot(user_content_profile, movie_features_vector_norm)
            
            # Scale the similarity (0 to 1) by GENRE_BOOST (1.0)
            similarity_boost = np.clip(similarity, 0, 1) * GENRE_BOOST


        # Total Boost: Combine the explicit genre match and the feature similarity score
        total_boost = genre_boost + similarity_boost
            
        # Add boost to the predicted score
        final_scores.append((mid, score + total_boost))

    # --- 4. Sort and Enrich Results ---
    top_preds = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_n]
        
    enriched = []
    for mid, final_score in top_preds:
        row_df = movies_df[movies_df["movieid"] == mid]
        if row_df.empty: continue
        row = row_df.iloc[0]
            
        movie_ratings = ratings_df[ratings_df["movieid"] == mid]["rating"]
        avg_rating = movie_ratings.mean() if not movie_ratings.empty else None
        tags = tags_df[tags_df["movieid"] == mid]["tag"].value_counts().head(3).index.tolist()
        genres_output = row["genres"].split("|") 
        
        # Fetch the poster URL
        poster_path = fetch_movie_poster(row["title"])
            
        enriched.append({
            "movieId": int(mid),
            "title": movie_lookup.get(mid, "Unknown"),
            "avg_rating": avg_rating,
            "genres": genres_output,
            "top_tags": tags,
            "poster_path": poster_path
        })
    return enriched

def fetch_movie_poster(movie_title: str) -> str:
    """Fetches the poster path for a movie title from TMDB, using a fallback on failure."""
    if not TMDB_API_KEY:
        print("⚠️ TMDB API Key is missing. Returning fallback.")
        return ""
    
    # Clean the movie title for better search accuracy
    cleaned_title = re.sub(r'\s*\([^)]*\)', '', movie_title).strip()

    try:
        # 1. Search for the movie by title
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {TMDB_API_KEY}"
        }

        params = {
            "query": cleaned_title,
            "include_adult": "false",
            "language": "en-US",
            "page": 1
        }

        response = requests.get(TMDB_BASE_URL, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        # 2. Extract the poster path from the top result
        if data and data.get('results'):
            first_result = data['results'][0]
            poster_path = first_result.get('poster_path')
            
            if poster_path:
                return f"{TMDB_POSTER_BASE}{poster_path}"

    except requests.RequestException as e:
        print(f"Error fetching poster for '{movie_title}': {e}")
        
    # 3. Return fallback if no poster is found, or if an error occurred during the request
    return ""

# -----------------------
# Pydantic Models
# -----------------------
class RecommendRequest(BaseModel):
    username: str
    liked_genres: List[str] = [] 
    liked_movies: List[str] = []
    top_n: int = 12

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
            liked_movies=req.liked_movies,
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
        
        # 1. If interaction is 'not_interested', we add it permanently and remove any previous interaction
        if req.interaction == 'not_interested':
            # Remove any existing interaction for this movie/user
            df = df[~((df["user_id"]==user_id)&(df["movie_id"]==req.movie_id))] 
            # Add the permanent 'not_interested' flag
            new_row = {"user_id": user_id, "movie_id": req.movie_id, "interaction": req.interaction}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 2. Handle 'remove' for Interested/Watched
        elif req.interaction == 'remove':
            df = df[~((df["user_id"]==user_id)&(df["movie_id"]==req.movie_id))]
            
        # 3. Handle 'interested' or 'watched'
        else: 
            # Remove any existing interaction first (toggle logic)
            df = df[~((df["user_id"]==user_id)&(df["movie_id"]==req.movie_id))]
            new_row = {"user_id": user_id,"movie_id": req.movie_id,"interaction": req.interaction}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
        df.to_csv(INTERACTIONS_FILE, index=False)
    return {"status":"success"}

@app.get("/users/{username}/history")
def get_history(username: str):
    user_id = get_or_create_user(username)
    history = []

    is_new_user = True
    interactions_df = pd.DataFrame(columns=["user_id", "movie_id", "interaction"]) 
    
    # 1. Load interactions file safely
    if os.path.exists(INTERACTIONS_FILE) and os.path.getsize(INTERACTIONS_FILE) > 0:
        try:
            interactions_df = pd.read_csv(INTERACTIONS_FILE)
        except pd.errors.EmptyDataError:
            pass 
    
    user_rows = interactions_df[interactions_df["user_id"] == user_id]
    
    # 2. Determine is_new_user and build history list
    if not user_rows.empty:
        # User is NOT new if they have ANY rows in the interaction file
        is_new_user = False 
        
        if movies_df is not None:
            for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"]):
                # Skip 'not_interested' in history display
                if inter == 'not_interested':
                    continue 
                row = movies_df[movies_df["movieid"] == mid]
                if not row.empty:
                    row = row.iloc[0]
                    history.append({
                        "movieId": int(mid),
                        "title": movie_lookup.get(mid, "Unknown"),
                        "interaction": inter,
                        "genres": row["genres"].split("|")
                    })

    # Note: If a user exists but has only 'not_interested' movies, is_new_user is False, 
    # and history will be empty (correct behavior).
    return {"history": history, "is_new_user": is_new_user}

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
# Helper endpoints for frontend checkboxes
# -----------------------
@app.get("/genres")
def get_genres():
    if movies_df is None:
        load_data()
        
    if movies_df is None or tags_df is None:
        raise HTTPException(status_code=503, detail="Server data not yet available.")
        
    # Standardize and count genre frequency
    all_genres_list = movies_df["genres"].str.split("|").explode().str.title().dropna()

    # Count frequency and get top genres
    genre_counts = all_genres_list.value_counts()
    top_genres = genre_counts.head(TOP_GENRE_COUNT).index.tolist()

    # Remove no genre option
    if "(No Genres Listed)" in top_genres:
        top_genres.remove("(No Genres Listed)")

    # Sort and return
    genres_to_return = sorted(top_genres)

    return genres_to_return

@app.get("/movies")
def get_top_movies():
    if movies_df is None or ratings_df is None:
        load_data()
        
    if movies_df is None or ratings_df is None:
        raise HTTPException(status_code=503, detail="Movie or Rating data not available.")
    
    # ----------------------------------------------------
    # Weighted Rating Logic (Bayesian Average)
    # ----------------------------------------------------
    
    movie_stats = ratings_df.groupby("movieid")["rating"].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movieid', 'v', 'R']
    
    m = movie_stats['v'].quantile(0.60) 
    C = movie_stats['R'].mean()
    
    qualified_movies = movie_stats[movie_stats['v'] >= m].copy()
    
    if qualified_movies.empty:
        print("⚠️ Not enough movies meet the 60th percentile vote threshold. Falling back to simple average ranking.")
        return get_top_movies_simple_fallback()
        
    def weighted_rating(row):
        v = row['v']
        R = row['R']
        return (v / (v + m) * R) + (m / (v + m) * C)
        
    qualified_movies['weighted_rating'] = qualified_movies.apply(weighted_rating, axis=1)
    
    qualified_movies = qualified_movies.sort_values('weighted_rating', ascending=False).head(TOP_MOVIE_COUNT)
    
    final_list = pd.merge(qualified_movies, movies_df[['movieid', 'title']], on='movieid', how='left')
    
    return [title for title in final_list["title"].tolist()]

def get_top_movies_simple_fallback():
    """Fallback if Bayesian method finds too few qualified movies."""
    avg_ratings = ratings_df.groupby("movieid")["rating"].mean()
    movies_copy = movies_df.copy()
    movies_copy["avg_rating"] = movies_copy["movieid"].map(avg_ratings).fillna(0)
    top_movies = movies_copy.sort_values("avg_rating", ascending=False).head(TOP_MOVIE_COUNT)
    return [title for title in top_movies["title"].tolist()]

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)