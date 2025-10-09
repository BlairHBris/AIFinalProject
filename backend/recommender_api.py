import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
MOVIES_FILE_ID = "1MaQ2UMPcKaOlLAW1W9CJ_CUrf4stHxDB"
RATINGS_FILE_ID = "1GYyvuhMTfYzz6ua8RkkuINILSsIiwLwe"

download_from_gdrive(MOVIES_FILE_ID, MOVIES_FILE)
download_from_gdrive(RATINGS_FILE_ID, RATINGS_FILE)
print("CSV files ready.")

# =======================
# 2. FastAPI Setup
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
# 3. File paths
# =======================
MODEL_FILE = "torch_recommender.pt"
USERS_FILE = "users.csv"
INTERACTIONS_FILE = "user_interactions.csv"
UPDATE_THRESHOLD = 10

# =======================
# 4. Load CSVs
# =======================
movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)

# Normalize column names
movies.columns = [c.strip().lower() for c in movies.columns]
ratings.columns = [c.strip().lower() for c in ratings.columns]

# Ensure expected columns exist
movies["genres"] = movies.get("genres", "").fillna("")
if "actors" not in movies.columns:
    movies["actors"] = ""

# One-hot encode genres
unique_genres = set("|".join(movies["genres"]).split("|"))
for g in unique_genres:
    if g:
        movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0)

movie_lookup = movies.set_index("movieid")["title"].to_dict()

# =======================
# 5. PyTorch Model
# =======================
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.movie_embed = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_idx, movie_idx):
        x = self.user_embed(user_idx) * self.movie_embed(movie_idx)
        x = self.fc(x)
        return x.squeeze()

# Map user/movie IDs
user_ids = ratings["userid"].unique()
movie_ids = ratings["movieid"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}
num_users, num_movies = len(user2idx), len(movie2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderNet(num_users, num_movies).to(device)

# =======================
# 5a. Train if model missing
# =======================
if not os.path.exists(MODEL_FILE):
    print("Training PyTorch model...")
    class RatingsDataset(Dataset):
        def __init__(self, df):
            self.users = torch.tensor(df["userid"].map(user2idx).values, dtype=torch.long)
            self.movies = torch.tensor(df["movieid"].map(movie2idx).values, dtype=torch.long)
            self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)
        def __len__(self):
            return len(self.ratings)
        def __getitem__(self, idx):
            return self.users[idx], self.movies[idx], self.ratings[idx]

    dataset = RatingsDataset(ratings)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(3):  # Increase for better accuracy
        total_loss = 0
        for u, m, r in loader:
            u, m, r = u.to(device), m.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, m)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/3, Loss: {total_loss/len(loader):.4f}")

    torch.save({"model_state_dict": model.state_dict(),
                "user2idx": user2idx,
                "movie2idx": movie2idx}, MODEL_FILE)
    print("Saved torch_recommender.pt")
else:
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    user2idx = checkpoint["user2idx"]
    movie2idx = checkpoint["movie2idx"]

model.eval()

# =======================
# 6. Ensure CSVs exist
# =======================
for f, cols in [(INTERACTIONS_FILE, ["user_id", "movie_id", "interaction"]),
                (USERS_FILE, ["user_id", "username"])]:
    if not os.path.exists(f):
        pd.DataFrame(columns=cols).to_csv(f, index=False)

# =======================
# 7. Pydantic Models
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
# 8. User Management
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

# =======================
# 9. Recommendation functions
# =======================
def predict_rating(user_id, movie_id):
    if user_id not in user2idx or movie_id not in movie2idx:
        return 3.5
    u = torch.tensor([user2idx[user_id]], device=device)
    m = torch.tensor([movie2idx[movie_id]], device=device)
    with torch.no_grad():
        return model(u, m).item()

def recommend_collab(user_id, top_n=10):
    rated = set(ratings[ratings["userid"] == user_id]["movieid"])
    candidates = [mid for mid in movie_lookup if mid not in rated]
    preds = [(mid, predict_rating(user_id, mid)) for mid in candidates]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": mid, "title": movie_lookup[mid]} for mid, _ in top_preds]

def recommend_content(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = set(ratings[ratings["userid"] == user_id]["movieid"])
    candidates = movies[~movies["movieid"].isin(rated)].copy()
    if liked_genres:
        candidates = candidates[candidates[liked_genres].sum(axis=1) > 0]
    if liked_actors:
        candidates = candidates[candidates["actors"].apply(lambda x: any(a in x for a in liked_actors))]
    top_candidates = candidates.sample(min(top_n, len(candidates)))
    return [{"movieId": row.movieid, "title": row.title} for _, row in top_candidates.iterrows()]

def recommend_hybrid(user_id, liked_genres=[], liked_actors=[], top_n=10):
    rated = set(ratings[ratings["userid"] == user_id]["movieid"])
    candidates = movies[~movies["movieid"].isin(rated)].copy()
    if liked_genres:
        candidates = candidates[candidates[liked_genres].sum(axis=1) > 0]
    if liked_actors:
        candidates = candidates[candidates["actors"].apply(lambda x: any(a in x for a in liked_actors))]
    preds = [(mid, predict_rating(user_id, mid)) for mid in candidates["movieid"]]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"movieId": mid, "title": movie_lookup[mid]} for mid, _ in top_preds]

# =======================
# 10. Feedback / retraining
# =======================
def retrain_if_needed():
    df = pd.read_csv(INTERACTIONS_FILE)
    if len(df) % UPDATE_THRESHOLD == 0 and len(df) > 0:
        print("Retraining not implemented yet. Use train_torch.py for now.")

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
    return {"history": [{"movieId": mid, "title": movie_lookup.get(mid, "Unknown"), "interaction": inter}
                        for mid, inter in zip(user_rows["movie_id"], user_rows["interaction"])]}
