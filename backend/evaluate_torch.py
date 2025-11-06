# evaluate_torch.py
"""
Evaluation script for the PyTorch Recommender Model.

Calculates:
1. RMSE (Root Mean Square Error) for rating prediction accuracy.
2. Precision@K (P@K) for ranking effectiveness.
3. Normalized Discounted Cumulative Gain@K (NDCG@K) for ranking quality, considering item position.

Runs evaluation in three modes: Collaborative Filtering (CF), Content-Based, and Hybrid.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------
# Config
# -----------------------
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
TAGS_FILE = "tags.csv"
MODEL_FILE = "torch_recommender.pt"

# Evaluation Hyperparameters
K = 10  # Number of recommendations to consider for P@K and NDCG@K
RELEVANCE_THRESHOLD = 3.5  # Rating >= 3.5 means the user "liked" the movie (relevant)
TEST_SIZE = 0.2  # Fraction of ratings to use for the test set
EMBEDDING_DIM = 32

# Determine device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature constants (must align with train_torch.py)
TOP_GENRES = 20
TOP_TAGS_COUNT = 512

# -----------------------
# Model Definition (Refactored to support CF/Content/Hybrid modes)
# -----------------------
class RecommenderNet(nn.Module):
    """
    Hybrid Recommender Model supporting CF, Content, or Hybrid modes
    by conditionally enabling embedding and feature paths.
    """
    def __init__(self, num_users, num_movies, embedding_dim, movie_feat_dim, 
                 use_cf=True, use_content=True):
        super(RecommenderNet, self).__init__()
        
        self.use_cf = use_cf
        self.use_content = use_content

        # Embeddings (CF part)
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.movie_embed = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)
        
        # Determine input size for the final FC layer
        fc_input_size = 0
        if self.use_cf:
            # If using CF, the input is the element-wise product of two EMBEDDING_DIM vectors
            fc_input_size += embedding_dim
        if self.use_content:
            # If using content, the input includes the movie features
            fc_input_size += movie_feat_dim

        if fc_input_size == 0:
            # This should only happen if both flags are False
            # For the sub-models, we still need the size calculation for correct weight slicing
            pass # Allow initialization to fail for models with missing weight parts later

        self.fc = nn.Linear(fc_input_size, 1)

    def forward(self, user_idx, movie_idx, movie_feats):
        x_parts = []

        if self.use_cf:
            u = self.user_embed(user_idx)
            m = self.movie_embed(movie_idx)
            x_cf = u * m
            x_parts.append(x_cf)
        
        if self.use_content:
            x_parts.append(movie_feats)

        # Concatenate all active feature parts
        if not x_parts:
            # Should not happen if initialization flags are set correctly
            raise ValueError("No feature paths enabled in forward pass.")
            
        x = torch.cat(x_parts, dim=1)
        
        # Output a single predicted rating
        x = self.fc(x)
        return x.squeeze()

# -----------------------
# Dataset Class for Evaluation
# -----------------------
class MovieDataset(Dataset):
    def __init__(self, data, movie_features, user2idx, movie2idx):
        # Data columns are lowercase from the read_csv in evaluate_model()
        self.user_indices = torch.tensor(data['userId'].map(user2idx).values, dtype=torch.long)
        self.movie_indices = torch.tensor(data['movieId'].map(movie2idx).values, dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32)
        
        # Map movie indices to features
        self.movie_features = []
        for movie_id in data['movieId'].values:
            # Check if feature exists, otherwise use zeros
            features = movie_features.loc[movie_id].values if movie_id in movie_features.index else np.zeros(movie_features.shape[1])
            self.movie_features.append(features)
            
        self.movie_features = torch.tensor(np.stack(self.movie_features), dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.movie_indices[idx], self.movie_features[idx], self.ratings[idx]

# -----------------------
# Evaluation Metrics
# -----------------------

def ndcg_at_k(y_true, y_score, k):
    """Normalized Discounted Cumulative Gain (NDCG) @ K."""
    df = pd.DataFrame({'true': y_true, 'score': y_score})
    # Sort by predicted score
    df = df.sort_values(by='score', ascending=False).head(k)

    # Calculate DCG: DCG = sum( (2^relevance_score - 1) / log2(i + 1) )
    # Here, relevance is binary (0 or 1) based on the RELEVANCE_THRESHOLD
    relevance = (df['true'] >= RELEVANCE_THRESHOLD).astype(int)
    
    # Calculate DCG
    # We use log2(i + 2) because i is 0-indexed, and we want log2(rank + 1)
    # Ranks start at 1, so index i corresponds to rank i+1
    dcg = np.sum((2**relevance - 1) / np.log2(np.arange(len(relevance)) + 2))

    # Calculate IDCG (Ideal DCG)
    # Get the top K true relevant items
    ideal_relevance = np.sort((y_true >= RELEVANCE_THRESHOLD).astype(int))[::-1][:k]
    idcg = np.sum((2**ideal_relevance - 1) / np.log2(np.arange(len(ideal_relevance)) + 2))

    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(y_true, y_score, k):
    """Precision @ K."""
    df = pd.DataFrame({'true': y_true, 'score': y_score})
    # Sort by predicted score and take top K
    df = df.sort_values(by='score', ascending=False).head(k)
    
    # Count true positives in the top K
    true_positives = (df['true'] >= RELEVANCE_THRESHOLD).sum()
    
    return true_positives / k

# -----------------------
# Feature Engineering (MUST MATCH train_torch.py EXACTLY)
# -----------------------

def prepare_movie_features(movies_df, tags_df):
    """
    Re-creates the feature matrix based on genres and tags,
    matching the capitalization and naming conventions of train_torch.py.
    The final DataFrame is indexed by 'movieid'.
    """
    movies_df = movies_df.reset_index(drop=True) 
    movies_df.columns = [c.strip().lower() for c in movies_df.columns]
    tags_df.columns = [c.strip().lower() for c in tags_df.columns]

    # --- 1. Genre Processing (Matches train_torch.py) ---
    movies_df['genres_list'] = movies_df["genres"].str.split("|").fillna('')
    
    # Standardize and count frequency of genres, then select top N
    all_genres = movies_df["genres"].str.split("|").explode().str.upper().dropna()
    genre_counts = all_genres.value_counts()
    top_genres = genre_counts.head(TOP_GENRES).index.tolist()
    
    genre_data = {}
    
    # One-hot encode using the raw, uppercase genre name as the column name (no prefix)
    for g in top_genres:
        genre_data[g] = movies_df["genres"].apply(lambda x: 1 if g in x.upper() else 0).astype(np.float16)

    # CRITICAL FIX: Add the expected '(NO GENRES LISTED)' column
    genre_data['(NO GENRES LISTED)'] = movies_df["genres"].apply(lambda x: 1 if x.lower() == '(no genres listed)' else 0)

    # Index by 'movieid'
    genre_df = pd.DataFrame(genre_data, index=movies_df.index).fillna(0)
    genre_df['movieid'] = movies_df['movieid']
    genre_df = genre_df.set_index('movieid')
    
    # --- 2. Tag Processing (Matches train_torch.py) ---
    tags_df["tag"] = tags_df["tag"].fillna("").astype(str).str.lower()
    
    top_tags = tags_df["tag"].value_counts().head(TOP_TAGS_COUNT).index.tolist()
    tags_subset = tags_df[tags_df["tag"].isin(top_tags)]
    
    # pivot to multi-hot matrix, indexed by 'movieid'
    tag_matrix = tags_subset.pivot_table(index="movieid", columns="tag", aggfunc="size", fill_value=0)
    tag_matrix = tag_matrix.reindex(columns=top_tags, fill_value=0).astype(np.float16)
    
    # --- 3. Combine Features (Index Join is safest) ---
    # Both genre_df and tag_matrix are now indexed by 'movieid' (lowercase)
    movie_features = genre_df.join(tag_matrix, how='left').fillna(0)
    
    return movie_features


# -----------------------
# Core Evaluation Logic
# -----------------------
def run_evaluation_mode(model_mode, model, movie_features_df, test_ratings, user2idx, movie2idx):
    """Calculates RMSE, P@K, and NDCG@K for a given model instance and mode."""
    
    # Create zero feature matrix for CF-only mode
    if model_mode == 'Collaborative Filtering':
        zero_features_df = pd.DataFrame(
            np.zeros((len(movie_features_df), movie_features_df.shape[1])),
            index=movie_features_df.index,
            columns=movie_features_df.columns
        )
        # Use the zeroed feature matrix for the CF evaluation
        movie_features_df = zero_features_df
        
    print(f"\n--- Running Evaluation: {model_mode} ---")
    
    # --- RMSE Calculation ---
    test_dataset = MovieDataset(test_ratings.copy(), movie_features_df, user2idx, movie2idx)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for u, m, mf, r in test_loader:
            u, m, mf, r = u.to(DEVICE), m.to(DEVICE), mf.to(DEVICE), r.to(DEVICE)
            pred = model(u, m, mf)
            pred = torch.clamp(pred, min=1.0, max=5.0) 
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(r.cpu().numpy())
            
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    print(f"✅ RMSE (Root Mean Square Error): {rmse:.4f}")

    # --- P@K and NDCG@K Calculation ---
    # Reload ratings for ranking exclusions (must use full set to check against train)
    train_ratings = pd.read_csv(RATINGS_FILE)
    train_ratings.columns = [c.strip().lower() for c in train_ratings.columns]
    
    # Get training set exclusion list
    train_rated_movies = train_ratings.groupby('userid')['movieid'].apply(set).to_dict()

    test_users = test_ratings['userId'].unique()
    
    p_k_scores = []
    ndcg_k_scores = []
    
    # Get the list of movie IDs that are indexed in the model AND have features
    all_movie_ids = list(movie2idx.keys())[1:] # Skip index 0 (padding)
    movies_with_features = [mid for mid in all_movie_ids if mid in movie_features_df.index]
    
    # Use the same test set rating mapping for ground truth
    true_ratings_map_test = test_ratings.set_index('movieId')['rating'].to_dict()

    # Optimization: Process all candidate movies in one batch per user
    for user_id in test_users:
        user_idx = user2idx.get(user_id)
        if user_idx is None:
            continue
            
        # 1. Identify movies rated in training (for exclusion)
        rated_in_train = train_rated_movies.get(user_id, set())
        
        # 2. Filter the pre-fetched list to only include candidate movies (not rated in train)
        candidate_movies_mids = [mid for mid in movies_with_features if mid not in rated_in_train]
        
        if not candidate_movies_mids or len(candidate_movies_mids) < K:
            continue
            
        # 3. Create prediction batch: indices and features
        candidate_indices = torch.tensor([movie2idx[mid] for mid in candidate_movies_mids], dtype=torch.long).to(DEVICE)
        
        # Pull features for only the candidate movies (VECTORIZED LOOKUP: FAST)
        candidate_features_df = movie_features_df.loc[candidate_movies_mids]
        candidate_features = torch.tensor(candidate_features_df.values, dtype=torch.float32).to(DEVICE)
        
        # Create user index tensor
        user_indices = torch.full((len(candidate_indices),), user_idx, dtype=torch.long).to(DEVICE)

        # 4. Predict ratings
        with torch.no_grad():
            predicted_ratings = model(user_indices, candidate_indices, candidate_features).cpu().numpy()

        # 5. Create ground truth for the *ranked list*
        y_true_rank = np.array([true_ratings_map_test.get(mid, 0.0) for mid in candidate_movies_mids])
        y_score_rank = predicted_ratings

        # 6. Calculate Metrics
        if (y_true_rank >= RELEVANCE_THRESHOLD).sum() > 0:
            p_k_scores.append(precision_at_k(y_true_rank, y_score_rank, K))
            ndcg_k_scores.append(ndcg_at_k(y_true_rank, y_score_rank, K))


    if p_k_scores:
        avg_p_k = np.mean(p_k_scores)
        avg_ndcg_k = np.mean(ndcg_k_scores)
        print(f"Users Evaluated for Ranking: {len(p_k_scores)}")
        print(f"✅ P@{K} (Precision at {K}): {avg_p_k:.4f}")
        print(f"✅ NDCG@{K} (Normalized Discounted Cumulative Gain at {K}): {avg_ndcg_k:.4f}")
    else:
        # Prevent errors if no users could be evaluated
        avg_p_k = 0.0
        avg_ndcg_k = 0.0
        print("⚠️ No users found for ranking evaluation.")
        
    return rmse, avg_p_k, avg_ndcg_k


def evaluate_model():
    print("--- Starting Model Evaluation ---")
    
    # 1. Load Data
    try:
        ratings_df = pd.read_csv(RATINGS_FILE)
        movies_df = pd.read_csv(MOVIES_FILE)
        tags_df = pd.read_csv(TAGS_FILE)
        
        ratings_df.columns = [c.strip().lower() for c in ratings_df.columns]
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        return

    # 2. Recreate Movie Features 
    movie_features_superset = prepare_movie_features(movies_df.copy(), tags_df.copy())
    
    # 3. Load Checkpoint & Prepare Final Features
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
        user2idx = checkpoint['user2idx']
        movie2idx = checkpoint['movie2idx']
        content_cols = checkpoint['content_cols'] 
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return
    
    try:
        missing_cols = [col for col in content_cols if col not in movie_features_superset.columns]
        if missing_cols:
            raise KeyError(f"The following feature columns are missing from the recreated features: {missing_cols}")

        movie_features_df = movie_features_superset.reindex(columns=content_cols).fillna(0)
    except KeyError as e:
        print(f"CRITICAL ERROR: Features from checkpoint not found in calculated features. Missing columns: {e}")
        return
        
    print(f"Movie Feature Dimension: {len(content_cols)}")
    
    # 4. Data Split
    ratings_df['mapped_movie'] = ratings_df['movieid'].map(movie2idx)
    ratings_df = ratings_df.dropna(subset=['mapped_movie'])
    ratings_df['mapped_movie'] = ratings_df['mapped_movie'].astype(int)

    try:
        train_ratings, test_ratings = train_test_split(
            ratings_df, 
            test_size=TEST_SIZE, 
            random_state=42, 
            stratify=ratings_df['userid']
        )
    except ValueError:
         train_ratings, test_ratings = train_test_split(
            ratings_df, 
            test_size=TEST_SIZE, 
            random_state=42
        )

    print(f"Total Ratings: {len(ratings_df)}")
    print(f"Test Ratings for RMSE: {len(test_ratings)}")
    
    # Rename for consistency with Dataset class
    test_ratings = test_ratings.rename(columns={'userid': 'userId', 'movieid': 'movieId'})
    
    num_users = len(user2idx)
    num_movies = len(movie2idx)
    movie_feat_dim = len(content_cols)
    
    results = {}
    
    # --- Get Weights from Hybrid Checkpoint ---
    fc_hybrid_weight = checkpoint['model_state_dict']['fc.weight']
    fc_bias = checkpoint['model_state_dict']['fc.bias']
    
    # Truncate the FC weight: CF part is the first EMBEDDING_DIM columns
    fc_cf_weight = fc_hybrid_weight[:, :EMBEDDING_DIM]
    # Truncate the FC weight: Content part is the rest of the columns
    fc_content_weight = fc_hybrid_weight[:, EMBEDDING_DIM:]


    # --- Mode 1: Hybrid (CF + Content) ---
    model_hybrid = RecommenderNet(num_users, num_movies, EMBEDDING_DIM, movie_feat_dim, use_cf=True, use_content=True).to(DEVICE)
    model_hybrid.load_state_dict(checkpoint['model_state_dict'])
    model_hybrid.eval()
    results['Hybrid'] = run_evaluation_mode('Hybrid', model_hybrid, movie_features_df.copy(), test_ratings, user2idx, movie2idx)
    
    
    # --- Mode 2: Collaborative Filtering Only (CF) ---
    model_cf = RecommenderNet(num_users, num_movies, EMBEDDING_DIM, movie_feat_dim, use_cf=True, use_content=False).to(DEVICE)
    
    cf_state_dict = {
        'user_embed.weight': checkpoint['model_state_dict']['user_embed.weight'],
        'movie_embed.weight': checkpoint['model_state_dict']['movie_embed.weight'],
        # Use the truncated CF-only weight for the final layer
        'fc.weight': fc_cf_weight,
        'fc.bias': fc_bias,
    }
    
    # Load state dict, ignoring the content-related layers that don't exist in the CF-only model
    model_cf.load_state_dict(cf_state_dict)
    model_cf.eval()
    
    # CF evaluation uses a zero feature matrix (done inside run_evaluation_mode)
    results['Collaborative Filtering'] = run_evaluation_mode('Collaborative Filtering', model_cf, movie_features_df.copy(), test_ratings, user2idx, movie2idx)

    
    # --- Mode 3: Content-Based Only (Content) ---
    # The Content-based model requires no user/movie embeddings to be instantiated
    model_content = RecommenderNet(num_users, num_movies, EMBEDDING_DIM, movie_feat_dim, use_cf=False, use_content=True).to(DEVICE)
    
    content_state_dict = {
        # Use the truncated Content-only weight for the final layer
        'fc.weight': fc_content_weight,
        'fc.bias': fc_bias,
    }
    
    # Load state dict, ignoring the embedding layers that don't exist in the Content-only model
    model_content.load_state_dict(content_state_dict, strict=False) 
    model_content.eval()
    
    # Content evaluation uses the full feature matrix
    results['Content-Based'] = run_evaluation_mode('Content-Based', model_content, movie_features_df.copy(), test_ratings, user2idx, movie2idx)
    
    # --- Final Results Table ---
    print("\n=============================================")
    print("      SUMMARY OF MODEL EVALUATION MODES      ")
    print("=============================================")
    print(f"{'Model Mode':<25}{'RMSE':<10}{f'P@{K}':<10}{f'NDCG@{K}':<10}")
    print("-" * 55)
    
    for mode, (rmse, p_k, ndcg_k) in results.items():
        # Handle cases where P@K or NDCG@K returned 0.0 due to no relevant items
        p_k_str = f"{p_k:.4f}" if p_k > 0 else "0.0000"
        ndcg_k_str = f"{ndcg_k:.4f}" if ndcg_k > 0 else "0.0000"
        
        print(f"{mode:<25}{rmse:<10.4f}{p_k_str:<10}{ndcg_k_str:<10}")
    print("=============================================")


if __name__ == "__main__":
    evaluate_model()