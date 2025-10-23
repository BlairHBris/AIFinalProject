# recommender_api.py
import os
import threading
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# =======================
# 0. Config
# =======================
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
TAGS_FILE = "tags.csv"

MOVIES_FILE_ID = "1MaQ2UMPcKaOlLAW1W9CJ_CUrf4stHxDB"
RATINGS_FILE_ID = "1GYyvuhMTfYzz6ua8RkkuINILSsIiwLwe"
TAGS_FILE_ID = "1XSEd3G5fFx5AF869QmoAYpjSTH_8Ztd8"

MODEL_FILE = "torch_recommender.pt"       # committed model
TMP_MODEL_FILE = "/tmp/torch_recommender.pt"  # ephemeral retrain

USERS_FILE = "users.csv"
INTERACTIONS_FILE = "user_interactions.csv"
UPDATE_THRESHOLD = 10

# =======================
# 1. Helpers: download from Google Drive
# =======================
def download_from_gdrive(file_id: str, dest_path: str):
    if os.path.exists(dest_path):
        return
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {dest_path} from Google Drive.")

download_from_gdrive(MOVIES_FILE_ID, MOVIES_FILE)
download_from_gdrive(RATINGS_FILE_ID, RATINGS_FILE)

# =======================
# 2. FastAPI setup
# =======================
app = FastAPI(title="Adaptive Movie Recommender API (PyTorch)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 3. Load CSVs and preprocess
# =======================
movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)

movies.columns = [c.strip().lower() for c in movies.columns]
ratings.columns = [c.strip().lower() for c in ratings.columns]

movies["genres"] = movies.get("genres", "").fillna("")
if "actors" not in movies.columns:
    movies["actors"] = ""

# genre one-hot
unique_genres = set("|".join(movies["genres"]).split("|"))
unique_genres = {g for g in unique_genres if g}
for g in unique_genres:
    movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0)

# tags
tag_cols = []
if os.path.exists(TAGS_FILE):
    tags = pd.read_csv(TAGS_FILE)
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
# 4. PyTorch model + dataset
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
num_users = len(user2idx)
num_movies = len(movie2idx)
num_features = len(content_cols)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderNet(num_users, num_movies, num_features).to(device)

class RatingsDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df.copy()
        self.df = self.df[self.df["userid"].isin(user2idx) & self.df["movieid"].isin(movie2idx)].reset_index(drop=True)
        self.users = torch.tensor(self.df["userid"].map(user2idx).values, dtype=torch.long)
        self.movies = torch.tensor(self.df["movieid"].map(movie2idx).values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32)
        self.features = torch.tensor(self.df[feature_cols].fillna(0).values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.features[idx], self.ratings[idx]

# =======================
# 5. Load or train initial model
# =======================
def train_initial_model(save_path=MODEL_FILE, epochs=3):
    dataset = RatingsDataset(ratings_full, content_cols)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for u, m, f, r in loader:
            u, m, f, r = u.to(device), m.to(device), f.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, m, f)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Initial train Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "user2idx": user2idx,
        "movie2idx": movie2idx
    }, save_path)
    print(f"Saved initial model to {save_path}")
    model.eval()

# Load checkpoint
load_file = TMP_MODEL_FILE if os.path.exists(TMP_MODEL_FILE) else MODEL_FILE if os.path.exists(MODEL_FILE) else None
if load_file:
    try:
        checkpoint = torch.load(load_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        user2idx = checkpoint["user2idx"]
        movie2idx = checkpoint["movie2idx"]
        print(f"Loaded model checkpoint from {load_file}")
    except Exception as e:
        print("Failed to load checkpoint; training initial model. Error:", e)
        train_initial_model()
else:
    print("No model checkpoint found; training initial model.")
    train_initial_model()

model.eval()

# =======================
# 6. Ensure CSV files exist
# =======================
for f, cols in [(INTERACTIONS_FILE, ["user_id", "movie_id", "interaction"]),
                (USERS_FILE, ["user_id", "username"])]:
    if not os.path.exists(f):
        pd.DataFrame(columns=cols).to_csv(f, index=False)

# =======================
# 7. Pydantic models
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

# =======================
# 8. User management
# =======================
def get_or_create_user(username: str) -> int:
    username = username.strip().lower()
    users_df = pd.read_csv(USERS_FILE)
    if username in users_df["username"].values:
        return int(users_df[users_df["username"] == username]["user_id"].values[0])
    new_id = int(users_df["user_id"].max() + 1) if not users_df.empty else 1
    users_df = pd.concat([users_df, pd.DataFrame([{"user_id": new_id, "username": username}])], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    return new_id

def get_user_rated_movies_appuser(user_id: int):
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
        return set(df[df["user_id"] == user_id]["movie_id"])
    return set()

# =======================
# 9. Recommendation logic
# =======================
def predict_rating(user_id, movie_id):
    if user_id not in user2idx or movie_id not in movie2idx:
        return 3.5
    u = torch.tensor([user2idx[user_id]], dtype=torch.long, device=device)
    m = torch.tensor([movie2idx[movie_id]], dtype=torch.long, device=device)
    try:
        feat_vals = movies.loc[movies["movieid"] == movie_id, content_cols].values[0]
    except Exception:
        feat_vals = [0.0] * len(content_cols)
    f = torch.tensor([feat_vals], dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(model(u, m, f).item())

def recommend_collab(user_id, top_n=10):
    rated = get_user_rated_movies_appuser(user_id)
    candidates = [mid for mid in movie_lookup.keys() if mid not in rated]
    preds = [(mid, predict_rating(user_id, mid)) for mid in candidates]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": int(mid), "title": movie_lookup.get(mid, "Unknown")} for mid, _ in top_preds]

def recommend_content(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = get_user_rated_movies_appuser(user_id)
    candidates = movies[~movies["movieid"].isin(rated)].copy()
    if liked_genres:
        existing = [g for g in liked_genres if g in candidates.columns]
        if existing:
            candidates = candidates[candidates[existing].sum(axis=1) > 0]
    if liked_actors:
        candidates = candidates[candidates["actors"].apply(lambda x: any(a.lower() in x.lower() for a in liked_actors) if isinstance(x, str) else False)]
    if candidates.empty:
        return []
    sampled = candidates.sample(min(top_n, len(candidates)))
    return [{"movieId": int(row.movieid), "title": row.title} for _, row in sampled.iterrows()]

def recommend_hybrid(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = get_user_rated_movies_appuser(user_id)
    candidates_df = movies[~movies["movieid"].isin(rated)].copy()
    if liked_genres:
        existing = [g for g in liked_genres if g in candidates_df.columns]
        if existing:
            candidates_df = candidates_df[candidates_df[existing].sum(axis=1) > 0]
    if liked_actors:
        candidates_df = candidates_df[candidates_df["actors"].apply(lambda x: any(a.lower() in x.lower() for a in liked_actors) if isinstance(x, str) else False)]
    if candidates_df.empty:
        return []

    preds = []
    # use model predictions for candidates if possible
    for mid in candidates_df["movieid"]:
        preds.append((mid, predict_rating(user_id, mid)))
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": int(mid), "title": movie_lookup.get(mid, "Unknown")} for mid, _ in top_preds]

# =======================
# 10. Background retraining
# =======================
retrain_lock = threading.Lock()

def retrain_if_needed():
    try:
        df = pd.read_csv(INTERACTIONS_FILE)
    except Exception:
        return
    if len(df) >= UPDATE_THRESHOLD and len(df) % UPDATE_THRESHOLD == 0:
        t = threading.Thread(target=_background_retrain, args=(df.copy(),), daemon=True)
        t.start()

def _background_retrain(interactions_df: pd.DataFrame):
    if not retrain_lock.acquire(blocking=False):
        print("Retrain already running — skipping this trigger.")
        return
    try:
        print("Starting background retraining...")
        base_ratings = pd.read_csv(RATINGS_FILE)
        interactions = interactions_df.copy()

        def map_interaction_to_rating(x):
            s = str(x).lower()
            if "like" in s or "love" in s or "favorite" in s:
                return 4.0
            if "view" in s or "watch" in s:
                return 3.5
            if "skip" in s:
                return 2.0
            return 3.5

        interactions["rating"] = interactions["interaction"].apply(map_interaction_to_rating)
        MAX_ML_USER = int(base_ratings["userid"].max()) + 1 if not base_ratings.empty else 100000
        interactions["userid"] = interactions["user_id"].astype(int) + MAX_ML_USER
        interactions = interactions.rename(columns={"movie_id": "movieid", "userid": "userid"})[["userid", "movieid", "rating"]]

        combined = pd.concat([base_ratings[["userid", "movieid", "rating"]], interactions], ignore_index=True)
        combined_full = combined.merge(movies[["movieid"] + content_cols], on="movieid", how="left").fillna(0)

        new_user_ids = combined_full["userid"].unique()
        new_movie_ids = combined_full["movieid"].unique()
        new_user2idx = {u: i for i, u in enumerate(new_user_ids)}
        new_movie2idx = {m: i for i, m in enumerate(new_movie_ids)}

        embedding_dim = model.user_embed.embedding_dim if hasattr(model, "user_embed") else 50
        new_model = RecommenderNet(len(new_user2idx), len(new_movie2idx), len(content_cols), embedding_dim=embedding_dim).to(device)

        combined_full["user_idx"] = combined_full["userid"].map(new_user2idx)
        combined_full["movie_idx"] = combined_full["movieid"].map(new_movie2idx)

        class RebuildDataset(Dataset):
            def __init__(self, df, feature_cols):
                self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
                self.movies = torch.tensor(df["movie_idx"].values, dtype=torch.long)
                self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
                self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)
            def __len__(self):
                return len(self.ratings)
            def __getitem__(self, i):
                return self.users[i], self.movies[i], self.features[i], self.ratings[i]

        ds = RebuildDataset(combined_full, content_cols)
        loader = DataLoader(ds, batch_size=512, shuffle=True)
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(new_model.parameters(), lr=0.01)

        new_model.train()
        epochs = 2
        for epoch in range(epochs):
            total_loss = 0.0
            for u, m, f, r in loader:
                u, m, f, r = u.to(device), m.to(device), f.to(device), r.to(device)
                opt.zero_grad()
                pred = new_model(u, m, f)
                loss = criterion(pred, r)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"Retrain epoch {epoch+1}/{epochs}, loss {total_loss/len(loader):.4f}")

        torch.save({
            "model_state_dict": new_model.state_dict(),
            "user2idx": new_user2idx,
            "movie2idx": new_movie2idx
        }, TMP_MODEL_FILE)

        global model, user2idx, movie2idx
        model = new_model
        user2idx = new_user2idx
        movie2idx = new_movie2idx
        model.eval()
        print("Retraining complete.")
    except Exception as e:
        print("Error during retraining:", e)
    finally:
        retrain_lock.release()

# =======================
# 11. API routes
# =======================
@app.post("/recommend/{rec_type}")
def recommend(rec_type: str, req: RecommendRequest):
    user_id = get_or_create_user(req.username)
    if rec_type == "collab":
        recs = recommend_collab(user_id, req.top_n)
    elif rec_type == "content":
        recs = recommend_content(user_id, req.liked_genres, req.liked_actors, req.top_n)
    elif rec_type == "hybrid":
        recs = recommend_hybrid(user_id, req.liked_genres, req.liked_actors, req.top_n)
    else:
        raise HTTPException(status_code=400, detail="Invalid recommendation type")
    return {"recommendations": recs}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    user_id = get_or_create_user(req.username)
    df = pd.read_csv(INTERACTIONS_FILE)
    new_row = {"user_id": int(user_id), "movie_id": int(req.movie_id), "interaction": req.interaction}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(INTERACTIONS_FILE, index=False)
    retrain_if_needed()
    return {"status": "success", "message": "Feedback recorded. Retraining will run in background if threshold met."}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    if os.path.exists(INTERACTIONS_FILE):
        df = pd.read_csv(INTERACTIONS_FILE)
    else:
        df = pd.DataFrame(columns=["user_id", "movie_id", "interaction"])
    user_rows = df[df["user_id"] == user_id]
    return {"history": [{"movieId": int(mid), "title": movie_lookup.get(mid, "Unknown"), "interaction": inter}
                        for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"])]}