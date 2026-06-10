import os
import pandas as pd
from recommender import ContentRecommender, CollaborativeRecommender, HybridRecommender

DATA_DIR = r"C:\Users\HP\Documents\KAIF\Slash\TASK 6\data"
SPOTIFY_DATA_PATH = os.path.join(DATA_DIR, "SpotifyFeatures.csv")
USER_DATA_PATH = os.path.join(DATA_DIR, "user_interactions.csv")

def main():
    print("=== STARTING RECOMMENDATION ENGINE VERIFICATION ===")
    
    # 1. Load data
    print("Loading datasets...")
    df_tracks = pd.read_csv(SPOTIFY_DATA_PATH)
    df_interactions = pd.read_csv(USER_DATA_PATH)
    
    # 2. Instantiate Content Recommender
    print("\n--- Testing Content-Based Recommender ---")
    content_rec = ContentRecommender(df_tracks)
    
    # Query song
    query_song = "Don't Let Me Be Lonely Tonight"
    query_artist = "Joseph Williams"
    print(f"Querying similar songs to: '{query_song}' by '{query_artist}'")
    
    content_recs = content_rec.get_recommendations(query_song, query_artist, top_n=5)
    print("\nContent-Based Recommendations:")
    print(content_recs.to_string())
    
    # 3. Instantiate Collaborative Recommender
    print("\n--- Testing Collaborative Recommender ---")
    collab_rec = CollaborativeRecommender(df_interactions, df_tracks)
    
    # Query User
    query_user = "USR_0001"
    print(f"Querying recommendations for user: {query_user}")
    
    collab_recs = collab_rec.get_recommendations(query_user, top_n=5)
    print("\nCollaborative Filtering Recommendations:")
    print(collab_recs.to_string())
    
    # 4. Instantiate Hybrid Recommender
    print("\n--- Testing Hybrid Recommender ---")
    hybrid_rec = HybridRecommender(content_rec, collab_rec)
    
    hybrid_recs = hybrid_rec.get_recommendations(
        user_id=query_user, 
        last_track_name=query_song, 
        last_artist_name=query_artist, 
        top_n=5, 
        content_weight=0.6
    )
    print("\nHybrid Recommendations:")
    print(hybrid_recs.to_string())
    
    print("\n=== VERIFICATION COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
