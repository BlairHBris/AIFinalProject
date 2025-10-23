import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests

# ==========================
# 1. Config / file IDs
# ==========================
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
TAGS_FILE = "tags.csv"

MOVIES_FILE_ID = "1MaQ2UMPcKaOlLAW1W9CJ_CUrf4stHxDB"
RATINGS_FILE_ID = "1GYyvuhMTfYzz6ua8RkkuINILSsIiwLwe"
TAGS_FILE_ID = "1XSEd3G5fFx5AF869QmoAYpjSTH_8Ztd8"

MODEL_FILE = "/tmp/torch_recommender.pt" if os.path.exists("/tmp") else "torch_recommender.pt"

# ==========================
# 2. Download helper
# ==========================
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

# Download CSVs
for fid, fname in [(MOVIES_FILE_ID, MOVIES_FILE),
                   (RATINGS_FILE_ID, RATINGS_FILE),
                   (TAGS_FILE_ID, TAGS_FILE)]:
    download_from_gdrive(fid, fname)

# ==========================
# 3. Load data
# ==========================
movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)
tags = pd.read_csv(TAGS_FILE)

movies.columns = [c.strip().lower() for c in movies.columns]
ratings.columns = [c.strip().lower() for c in ratings.columns]
tags.columns = [c.strip().lower() for c in tags.columns]

# Ensure columns
if "actors" not in movies.columns:
    movies["actors"] = ""
movies["genres"] = movies.get("genres", "").fillna("")

# ==========================
# 4. Process genres (multi-hot)
# ==========================
unique_genres = set("|".join(movies["genres"]).split("|"))
unique_genres = {g for g in unique_genres if g}  # remove empty
for g in unique_genres:
    movies[g] = movies["genres"].apply(lambda x: 1 if g in x else 0)

# ==========================
# 5. Process tags efficiently
# ==========================
tags["tag"] = tags["tag"].fillna("").str.lower()
top_tags = tags["tag"].value_counts().head(1000).index.tolist()
tags_subset = tags[tags["tag"].isin(top_tags)]

# pivot to multi-hot
tag_matrix = tags_subset.pivot_table(index="movieid", columns="tag", aggfunc="size", fill_value=0)
movies = movies.merge(tag_matrix, on="movieid", how="left").fillna(0)

# All feature columns for hybrid model
content_cols = sorted(list(unique_genres)) + list(top_tags)

# ==========================
# 6. Map users/movies
# ==========================
user_ids = ratings["userid"].unique()
movie_ids = ratings["movieid"].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"] = ratings["userid"].map(user2idx)
ratings["movie_idx"] = ratings["movieid"].map(movie2idx)

num_users = len(user2idx)
num_movies = len(movie2idx)

# Movie features tensor aligned to movie_idx
movie_features = movies.set_index("movieid").reindex(movie_ids)[content_cols].fillna(0)
movie_features_tensor = torch.tensor(movie_features.values, dtype=torch.float32)

# ==========================
# 7. Dataset
# ==========================
class RatingsDataset(Dataset):
    def __init__(self, df, movie_feats):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.movie_feats = movie_feats
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.movie_feats[self.movies[idx]], self.ratings[idx]

dataset = RatingsDataset(ratings, movie_features_tensor)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

# ==========================
# 8. Model
# ==========================
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, movie_feat_dim, embedding_dim=50):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.movie_embed = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim + movie_feat_dim, 1)

    def forward(self, user_idx, movie_idx, movie_feats):
        x = self.user_embed(user_idx) * self.movie_embed(movie_idx)
        x = torch.cat([x, movie_feats], dim=1)
        x = self.fc(x)
        return x.squeeze()

model = RecommenderNet(num_users, num_movies, movie_feat_dim=len(content_cols), embedding_dim=50)

# ==========================
# 9. Training
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 5

model.train()
for epoch in range(epochs):
    total_loss = 0
    for u, m, mf, r in loader:
        u, m, mf, r = u.to(device), m.to(device), mf.to(device), r.to(device)
        optimizer.zero_grad()
        pred = model(u, m, mf)
        loss = criterion(pred, r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# ==========================
# 10. Save model
# ==========================
torch.save({
    "model_state_dict": model.state_dict(),
    "user2idx": user2idx,
    "movie2idx": movie2idx,
    "content_cols": content_cols
}, MODEL_FILE)

print(f"Saved model to {MODEL_FILE}")
