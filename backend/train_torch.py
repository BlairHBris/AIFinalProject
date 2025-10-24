# train_torch.py
"""
Local training script. Produces 'torch_recommender.pt' for backend.

Key points:
- Uses ratings + tags + genres for hybrid recommendation.
- **Genres: Top 15 most frequent, capitalized** (matches final API feature set).
- Saves full checkpoint required by FastAPI: 
    { "model_state_dict": ..., "user2idx": ..., "movie2idx": ..., "content_cols": ... }
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
TAGS_FILE = "tags.csv"
MODEL_FILE = "torch_recommender.pt"

TOP_TAGS = 512
TOP_GENRES = 20 # <-- NEW CONSTANT for consistency
EMBEDDING_DIM = 32
BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Sanity checks
# -----------------------
for f in (MOVIES_FILE, RATINGS_FILE, TAGS_FILE):
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing required file: {f}. Run locally where CSVs exist.")

# -----------------------
# Load & preprocess
# -----------------------
movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)
tags = pd.read_csv(TAGS_FILE)

movies.columns = [c.strip().lower() for c in movies.columns]
ratings.columns = [c.strip().lower() for c in ratings.columns]
tags.columns = [c.strip().lower() for c in tags.columns]

# Ensure 'genres' column is processed for feature engineering
movies["genres"] = movies.get("genres", "").fillna("") 

# --- UPDATED GENRE PROCESSING: Top 15, Capitalized ---
# 1. Standardize and count frequency
all_genres = movies["genres"].str.split("|").explode().str.upper().dropna()
genre_counts = all_genres.value_counts()

# 2. Select top N genres
top_genres = genre_counts.head(TOP_GENRES).index.tolist()
unique_genres = set(top_genres)

# 3. One-hot encode only the top genres
for g in top_genres:
    # Match the API's capitalization and column naming
    movies[g] = movies["genres"].apply(
        lambda x: 1 if g in x.upper() else 0
    ).astype(np.float16)
# --- END UPDATED GENRE PROCESSING ---


# top tags (limit to 512)
tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
top_tags = tags["tag"].value_counts().head(TOP_TAGS).index.tolist()
tags_subset = tags[tags["tag"].isin(top_tags)]

# pivot to multi-hot matrix (dense)
tag_matrix = tags_subset.pivot_table(index="movieid", columns="tag", aggfunc="size", fill_value=0)
tag_matrix = tag_matrix.reindex(columns=top_tags, fill_value=0).astype(np.float16)
movies = movies.merge(tag_matrix, on="movieid", how="left").fillna(0)

# final content columns (genres + top tags)
content_cols = sorted(list(unique_genres)) + top_tags

# -----------------------
# User/movie mapping
# -----------------------
user_ids = ratings["userid"].unique()
movie_ids = ratings["movieid"].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"] = ratings["userid"].map(user2idx)
ratings["movie_idx"] = ratings["movieid"].map(movie2idx)

# precompute movie features aligned to movie_idx order
# Ensure indexing only uses columns present after feature creation
movie_features = movies.set_index("movieid").reindex(movie_ids)[content_cols].fillna(0).astype(np.float16)
movie_features_tensor = torch.tensor(movie_features.values, dtype=torch.float16)

# -----------------------
# Dataset
# -----------------------
class RatingsDataset(Dataset):
    def __init__(self, df, movie_feats):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.movie_feats = movie_feats
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        # Keeps main tensor in float16 for memory, converts just the batch slice to float32 for training stability
        feats = self.movie_feats[self.movies[idx]].to(torch.float32) 
        return self.users[idx], self.movies[idx], feats, self.ratings[idx]

dataset = RatingsDataset(ratings, movie_features_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# -----------------------
# Model
# -----------------------
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, movie_feat_dim, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.movie_embed = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim + movie_feat_dim, 1)

    def forward(self, user_idx, movie_idx, movie_feats):
        u = self.user_embed(user_idx)
        m = self.movie_embed(movie_idx)
        x = u * m
        x = torch.cat([x, movie_feats], dim=1)
        x = self.fc(x)
        return x.squeeze()

model = RecommenderNet(len(user2idx), len(movie2idx), movie_feat_dim=len(content_cols)).to(DEVICE)

# -----------------------
# Train
# -----------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for u, m, mf, r in loader:
        u, m, mf, r = u.to(DEVICE), m.to(DEVICE), mf.to(DEVICE), r.to(DEVICE)
        optimizer.zero_grad()
        pred = model(u, m, mf)
        loss = criterion(pred, r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(loader):.4f}")

# -----------------------
# Save checkpoint
# -----------------------
checkpoint = {
    "model_state_dict": model.state_dict(),
    "user2idx": user2idx,
    "movie2idx": movie2idx,
    "content_cols": content_cols
}
torch.save(checkpoint, MODEL_FILE)
print(f"Saved checkpoint to {MODEL_FILE}")