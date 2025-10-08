import os
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# =======================
# 1. Download CSVs from Google Drive
# =======================
def download_from_gdrive(file_id: str, dest_path: str):
    if os.path.exists(dest_path):
        return
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith('download_warning'):
            token = v

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {dest_path} from Google Drive.")

# File paths
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
MOVIES_FILE_ID = "1wuqyw7RS-9Qrpx6COfM-vKwOE3ta1U4w"
RATINGS_FILE_ID = "1WB8wHpB117BGg0pnP8W7xOUtegPIVy2C"

download_from_gdrive(MOVIES_FILE_ID, MOVIES_FILE)
download_from_gdrive(RATINGS_FILE_ID, RATINGS_FILE)
print("CSV files ready.")

# =======================
# 2. FastAPI app setup
# =======================
app = FastAPI(title="Adaptive Movie Recommender API (PyTorch, Google Drive CSVs)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 3. Paths & constants
# =======================
MODEL_FILE = "torch_recommender.pt"
INTERACTIONS_FILE = "user_interactions.csv"
USERS_FILE = "users.csv"
UPDATE_THRESHOLD = 10

# =======================
# 4. Load movie and ratings data
# =======================
movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)
movies["genres"] = movies["genres"].fillna("")
if "actors" not in movies.columns:
    movies["actors"] = ""

# One-hot encode genres
unique_genres = set("|".join(movies["genres"]).split("|"))
for g in unique_genres:
    if g:
        movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0)

movie_lookup = movies.set_index("movieId")["title"].to_dict()

# =======================
# 5. PyTorch model
# =======================
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.movie_embed = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_embed(user_idx)
        movie_vec = self.movie_embed(movie_idx)
        x = user_vec * movie_vec
        x = self.fc(x)
        return x.squeeze()

# Map users and movies to indices
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}
num_users, num_movies = len(user2idx), len(movie2idx)

# Instantiate model
model = RecommenderNet(num_users, num_movies)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load model if exists
if os.path.exists(MODEL_FILE):
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    user2idx = checkpoint["user2idx"]
    movie2idx = checkpoint["movie2idx"]
else:
    print("torch_recommender.pt not found. Train model using train_torch.py first.")

model.eval()

# =======================
# 6. Ensure CSVs exist
# =======================
for f, cols in [(INTERACTIONS_FILE, ["user_id", "movie_id", "interaction"]),
                (USERS_FILE, ["user_id", "username"])]:
    if not os.path.exists(f):
        pd.DataFrame(columns=cols).to_csv(f, index=False)

# =======================
# 7. Request models
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
    else:
        new_id = int(users_df["user_id"].max() + 1) if not users_df.empty else 1
        users_df = pd.concat([users_df, pd.DataFrame([{"user_id": new_id, "username": username}])], ignore_index=True)
        users_df.to_csv(USERS_FILE, index=False)
        return new_id

# =======================
# 9. Recommendation functions
# =======================
def predict_rating(user_id: int, movie_id: int) -> float:
    if user_id not in user2idx or movie_id not in movie2idx:
        return 3.5
    user_idx = torch.tensor([user2idx[user_id]], device=device)
    movie_idx = torch.tensor([movie2idx[movie_id]], device=device)
    with torch.no_grad():
        pred = model(user_idx, movie_idx).item()
    return pred

def recommend_collab(user_id: int, top_n=10):
    rated = set(ratings[ratings["userId"] == user_id]["movieId"])
    candidates = [mid for mid in movie_lookup if mid not in rated]
    preds = [(mid, predict_rating(user_id, mid)) for mid in candidates]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": mid, "title": movie_lookup[mid]} for mid, _ in top_preds]

def recommend_content(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = set(ratings[ratings["userId"] == user_id]["movieId"])
    candidates = movies[~movies["movieId"].isin(rated)].copy()
    if liked_genres:
        candidates = candidates[candidates[liked_genres].sum(axis=1) > 0]
    if liked_actors:
        candidates = candidates[candidates["actors"].apply(lambda x: any(a in x for a in liked_actors))]
    top_candidates = candidates.sample(min(top_n, len(candidates)))
    return [{"movieId": row.movieId, "title": row.title} for _, row in top_candidates.iterrows()]

def recommend_hybrid(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = set(ratings[ratings["userId"] == user_id]["movieId"])
    candidates = movies[~movies["movieId"].isin(rated)].copy()
    if liked_genres:
        candidates = candidates[candidates[liked_genres].sum(axis=1) > 0]
    if liked_actors:
        candidates = candidates[candidates["actors"].apply(lambda x: any(a in x for a in liked_actors))]
    preds = [(mid, predict_rating(user_id, mid)) for mid in candidates["movieId"]]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": mid, "title": movie_lookup[mid]} for mid, _ in top_preds]

# =======================
# 10. Feedback / retraining
# =======================
def retrain_if_needed():
    df = pd.read_csv(INTERACTIONS_FILE)
    if len(df) % UPDATE_THRESHOLD == 0 and len(df) > 0:
        print("Retraining not implemented for PyTorch model yet. Use train_torch.py to retrain.")

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
    new_row = {"user_id": user_id, "movie_id": req.movie_id, "interaction": req.interaction}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(INTERACTIONS_FILE, index=False)
    retrain_if_needed()
    return {"status": "success"}

@app.get("/users/{username}/history")
def get_user_history(username: str):
    user_id = get_or_create_user(username)
    df = pd.read_csv(INTERACTIONS_FILE)
    user_rows = df[df["user_id"] == user_id]
    results = [
        {"movieId": mid, "title": movie_lookup.get(mid, "Unknown"), "interaction": inter}
        for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"])
    ]
    return {"history": results}
