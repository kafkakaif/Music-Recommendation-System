import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = r"C:\Users\HP\Documents\KAIF\Slash\TASK 6\data"
SPOTIFY_DATA_PATH = os.path.join(DATA_DIR, "SpotifyFeatures.csv")
USER_DATA_PATH = os.path.join(DATA_DIR, "user_interactions.csv")
SPOTIFY_CSV_URL = "https://raw.githubusercontent.com/sushmaakoju/spotify-tracks-data-analysis/main/SpotifyFeatures.csv"

def download_dataset():
    """Download the Spotify Features dataset from Github if it doesn't exist."""
    if not os.path.exists(SPOTIFY_DATA_PATH):
        print(f"Downloading Spotify dataset from {SPOTIFY_CSV_URL}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        # Use simple urllib to retrieve the file
        urllib.request.urlretrieve(SPOTIFY_CSV_URL, SPOTIFY_DATA_PATH)
        print("Download complete!")
    else:
        print("Spotify dataset already exists locally.")

def clean_and_preprocess():
    """Load the Spotify features dataset, clean, and normalize columns."""
    print("Loading Spotify dataset...")
    df = pd.read_csv(SPOTIFY_DATA_PATH)
    
    # Check for missing values and fill/drop
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Found {missing} missing values. Cleaning...")
        df = df.dropna()
        
    print(f"Dataset loaded with {df.shape[0]} tracks and {df.shape[1]} features.")
    
    # Scale audio features for recommendation similarity
    numerical_features = [
        'popularity', 'acousticness', 'danceability', 
        'energy', 'instrumentalness', 'liveness', 
        'loudness', 'speechiness', 'tempo', 'valence'
    ]
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    print("Preprocessed numerical features successfully.")
    return df, df_scaled

def generate_synthetic_user_interactions(df_tracks, num_users=1500, num_interactions=40000):
    """
    Generate synthetic user listening history linked to the Spotify dataset.
    We bias the generation towards popular tracks to simulate real user behavior.
    """
    if os.path.exists(USER_DATA_PATH):
        print("User interactions dataset already exists locally.")
        return pd.read_csv(USER_DATA_PATH)
        
    print(f"Generating synthetic interaction dataset for {num_users} users...")
    
    np.random.seed(42)
    
    # Create user IDs
    user_ids = [f"USR_{str(i).zfill(4)}" for i in range(1, num_users + 1)]
    
    # Extract track IDs and popularity
    track_ids = df_tracks['track_id'].unique()
    
    # Map track IDs to their popularity for weighting (handle cases where same track has different popularities)
    track_popularity = df_tracks.groupby('track_id')['popularity'].max().reindex(track_ids).fillna(0).values
    
    # Normalize popularity to form a probability distribution for selection bias
    popularity_prob = track_popularity + 1 # avoid 0 probabilities
    popularity_prob = popularity_prob / popularity_prob.sum()
    
    interactions = []
    
    # For each user, generate a random number of interactions (between 15 and 50)
    for user_id in user_ids:
        # Determine number of songs this user listened to
        n_songs = np.random.randint(15, 50)
        
        # Select songs based on popularity weights
        selected_tracks = np.random.choice(track_ids, size=n_songs, replace=False, p=popularity_prob)
        
        for track_id in selected_tracks:
            # Generate random listen count (1 to 40)
            listen_count = int(np.random.negative_binomial(1, 0.1) + 1)
            # Clip listen count to a reasonable max
            listen_count = min(listen_count, 60)
            
            # Generate user rating (1 to 5 stars) based on listen count (higher listens tend to have higher ratings)
            if listen_count >= 15:
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif listen_count >= 5:
                rating = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
            else:
                rating = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.4, 0.3])
                
            interactions.append({
                'user_id': user_id,
                'track_id': track_id,
                'listen_count': listen_count,
                'rating': rating
            })
            
    df_interactions = pd.DataFrame(interactions)
    df_interactions.to_csv(USER_DATA_PATH, index=False)
    print(f"Generated {df_interactions.shape[0]} user interactions and saved to {USER_DATA_PATH}.")
    return df_interactions

if __name__ == "__main__":
    download_dataset()
    df_tracks, df_scaled = clean_and_preprocess()
    generate_synthetic_user_interactions(df_tracks)
    print("Data Preprocessing Step Completed Successfully!")
