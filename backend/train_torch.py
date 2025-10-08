import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================
# 1. Load Data
# ==========================
ratings = pd.read_csv("ratings.csv")  # userId, movieId, rating

# Create unique indices for users and movies
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"] = ratings["userId"].map(user2idx)
ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

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
