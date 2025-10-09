import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import requests


# ==========================
# 1. Load Data
# ==========================
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

ratings = pd.read_csv("ratings.csv")  # userId, movieId, rating

# Normalize column names (strip spaces and lowercase)
ratings.columns = [c.strip().lower() for c in ratings.columns]

# Map the expected columns
user_col = "userid" if "userid" in ratings.columns else "userId"
movie_col = "movieid" if "movieid" in ratings.columns else "movieId"
rating_col = "rating"

# Create unique indices for users and movies
user_ids = ratings[user_col].unique()
movie_ids = ratings[movie_col].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"] = ratings[user_col].map(user2idx)
ratings["movie_idx"] = ratings[movie_col].map(movie2idx)

num_users = len(user_ids)
num_movies = len(movie_ids)

# ==========================
# 2. Dataset Class
# ==========================
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

dataset = RatingsDataset(ratings)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

# ==========================
# 3. Model Definition
# ==========================
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.movie_embed = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_embed(user_idx)
        movie_vec = self.movie_embed(movie_idx)
        x = user_vec * movie_vec  # element-wise product
        x = self.fc(x)
        return x.squeeze()

model = RecommenderNet(num_users, num_movies, embedding_dim=50)

# ==========================
# 4. Training Setup
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==========================
# 5. Train Model
# ==========================
epochs = 5  # Increase for better accuracy
for epoch in range(epochs):
    total_loss = 0
    for u, m, r in loader:
        u, m, r = u.to(device), m.to(device), r.to(device)
        optimizer.zero_grad()
        pred = model(u, m)
        loss = criterion(pred, r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# ==========================
# 6. Save Model
# ==========================
torch.save({
    "model_state_dict": model.state_dict(),
    "user2idx": user2idx,
    "movie2idx": movie2idx
}, "torch_recommender.pt")

print("Saved torch_recommender.pt")