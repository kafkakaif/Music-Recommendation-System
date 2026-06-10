import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

class ContentRecommender:
    def __init__(self, df_tracks):
        self.df = df_tracks.copy().reset_index(drop=True)
        # Numerical audio features to use for similarity
        self.feature_cols = [
            'popularity', 'acousticness', 'danceability', 
            'energy', 'instrumentalness', 'liveness', 
            'loudness', 'speechiness', 'tempo', 'valence'
        ]
        self._prepare_features()
        
    def _prepare_features(self):
        """Normalize the numerical features and prepare genre one-hot encoding."""
        # Normalize audio features
        scaler = MinMaxScaler()
        self.scaled_features = scaler.fit_transform(self.df[self.feature_cols])
        
        # One-hot encode genres (weighted to give genre matches higher importance)
        genre_dummies = pd.get_dummies(self.df['genre'], prefix='genre').values
        self.genre_features = genre_dummies * 1.5  # Weight genre matching more heavily
        
        # Combine numerical and genre features into a single feature matrix
        self.feature_matrix = np.hstack([self.scaled_features, self.genre_features])
        
    def get_recommendations(self, track_name, artist_name=None, top_n=10):
        """Recommend songs similar to a given song name and artist (optional)."""
        # Find the query song
        if artist_name:
            matches = self.df[(self.df['track_name'].str.lower() == track_name.lower()) & 
                              (self.df['artist_name'].str.lower() == artist_name.lower())]
        else:
            matches = self.df[self.df['track_name'].str.lower() == track_name.lower()]
            
        if matches.empty:
            print(f"Track '{track_name}' not found in dataset.")
            return pd.DataFrame()
            
        # If multiple matches, take the most popular one
        query_idx = matches['popularity'].idxmax()
        query_vector = self.feature_matrix[query_idx].reshape(1, -1)
        
        # Compute cosine similarity between query song and all other songs
        similarities = cosine_similarity(query_vector, self.feature_matrix).flatten()
        
        # Get the top_n indices (excluding the query song itself)
        related_indices = np.argsort(similarities)[::-1]
        related_indices = [idx for idx in related_indices if idx != query_idx]
        
        # Get metadata for recommended songs
        recommendations = self.df.iloc[related_indices[:top_n]].copy()
        recommendations['similarity_score'] = similarities[related_indices[:top_n]]
        
        return recommendations[['track_id', 'track_name', 'artist_name', 'genre', 'popularity', 'similarity_score']]

class CollaborativeRecommender:
    def __init__(self, df_interactions, df_tracks):
        self.df_interactions = df_interactions.copy()
        self.df_tracks = df_tracks.copy()
        
        # Calculate user rating averages to center predictions
        self.user_means = self.df_interactions.groupby('user_id')['rating'].mean().to_dict()
        self.global_mean = self.df_interactions['rating'].mean()
        
        self.unique_users = sorted(self.df_interactions['user_id'].unique())
        self.unique_tracks = sorted(self.df_interactions['track_id'].unique())
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.unique_users)}
        self.track_to_idx = {tid: idx for idx, tid in enumerate(self.unique_tracks)}
        self.idx_to_track = {idx: tid for tid, idx in self.track_to_idx.items()}
        
        self.model = None
        self.user_features = None
        self.item_features = None
        
        self._fit_svd()
        
    def _fit_svd(self, n_components=25):
        """Fit Truncated SVD to the sparse user-item interaction matrix."""
        print("Fitting Collaborative Filtering model (SVD)...")
        
        # Centering ratings by subtracting user means to improve SVD quality
        rows = []
        cols = []
        centered_ratings = []
        
        for _, row in self.df_interactions.iterrows():
            uid = row['user_id']
            tid = row['track_id']
            rating = row['rating']
            
            rows.append(self.user_to_idx[uid])
            cols.append(self.track_to_idx[tid])
            # Center the rating
            centered_ratings.append(rating - self.user_means.get(uid, self.global_mean))
            
        # Create Sparse CSR Matrix
        self.interaction_matrix = csr_matrix(
            (centered_ratings, (rows, cols)), 
            shape=(len(self.unique_users), len(self.unique_tracks))
        )
        
        # Apply SVD
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_features = self.svd.fit_transform(self.interaction_matrix)
        self.item_features = self.svd.components_.T
        
        # Compute standard deviation of SVD predictions to scale them back appropriately
        # This helps in scaling raw dot products back to the 1-5 star rating variance
        raw_predictions = np.dot(self.user_features, self.svd.components_)
        self.prediction_std = np.std(raw_predictions) if np.std(raw_predictions) > 0 else 1.0
        
        print(f"SVD decomposition completed. Explained variance: {self.svd.explained_variance_ratio_.sum():.4f}")
        
    def predict_rating(self, user_id, track_id):
        """Predict user rating for a specific track, adding back user mean."""
        user_mean = self.user_means.get(user_id, self.global_mean)
        
        if user_id not in self.user_to_idx or track_id not in self.track_to_idx:
            # Cold start fallback
            return round(user_mean, 1)
            
        u_idx = self.user_to_idx[user_id]
        t_idx = self.track_to_idx[track_id]
        
        # Dot product gives predicted rating offset (centered)
        raw_pred = np.dot(self.user_features[u_idx], self.svd.components_[:, t_idx])
        
        # Scale residual to fit standard rating distribution
        scaled_residual = (raw_pred / self.prediction_std) * 0.8
        
        # Add back user mean
        predicted_rating = user_mean + scaled_residual
        return np.clip(round(float(predicted_rating), 1), 1.0, 5.0)
        
    def get_recommendations(self, user_id, top_n=10):
        """Recommend songs for a user that they haven't listened to yet."""
        user_mean = self.user_means.get(user_id, self.global_mean)
        
        if user_id not in self.user_to_idx:
            # Cold start: Return popular tracks
            popular_tracks = self.df_tracks.sort_values(by='popularity', ascending=False).head(top_n)
            rec_df = popular_tracks[['track_id', 'track_name', 'artist_name', 'genre', 'popularity']].copy()
            rec_df['predicted_rating'] = round(self.global_mean, 1)
            return rec_df
            
        u_idx = self.user_to_idx[user_id]
        
        # Predict rating residuals for ALL tracks
        raw_residuals = np.dot(self.user_features[u_idx], self.svd.components_)
        
        # Scale and add user mean
        predicted_ratings = user_mean + (raw_residuals / self.prediction_std) * 0.8
        predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)
        
        # Find tracks the user has already listened to
        user_listened_tracks = self.df_interactions[self.df_interactions['user_id'] == user_id]['track_id'].values
        user_listened_indices = [self.track_to_idx[tid] for tid in user_listened_tracks if tid in self.track_to_idx]
        
        # Mask already listened songs by setting predicted rating to -1
        predicted_ratings[user_listened_indices] = -1.0
        
        # Get top indices
        top_indices = np.argsort(predicted_ratings)[::-1][:top_n]
        
        # Map back to track IDs and retrieve metadata
        rec_track_ids = [self.idx_to_track[idx] for idx in top_indices]
        
        # Filter metadata
        recommendations = self.df_tracks[self.df_tracks['track_id'].isin(rec_track_ids)].copy()
        
        # Add predicted rating score
        rating_map = {tid: predicted_ratings[self.track_to_idx[tid]] for tid in rec_track_ids}
        recommendations['predicted_rating'] = recommendations['track_id'].map(rating_map)
        
        # Clean rounding
        recommendations['predicted_rating'] = recommendations['predicted_rating'].round(1)
        
        return recommendations.sort_values(by='predicted_rating', ascending=False)[
            ['track_id', 'track_name', 'artist_name', 'genre', 'popularity', 'predicted_rating']
        ]

class HybridRecommender:
    def __init__(self, content_recommender, collab_recommender):
        self.content = content_recommender
        self.collab = collab_recommender
        self.df_tracks = content_recommender.df
        
    def get_recommendations(self, user_id, last_track_name, last_artist_name=None, top_n=10, content_weight=0.5):
        """
        Produce hybrid recommendations combining content-similarity (to last_track_name)
        and collaborative rating predictions for user_id.
        """
        # 1. Get Content Recommendations (wider list, say top 100)
        content_recs = self.content.get_recommendations(last_track_name, last_artist_name, top_n=100)
        
        if content_recs.empty:
            # If song not found, fall back to collaborative recommendations
            return self.collab.get_recommendations(user_id, top_n=top_n)
            
        # 2. Get Collaborative ratings for these content recommendations
        hybrid_scores = []
        
        for _, row in content_recs.iterrows():
            track_id = row['track_id']
            similarity = row['similarity_score']
            
            # Predict rating using Collaborative Filtering
            predicted_rating = self.collab.predict_rating(user_id, track_id)
            # Normalize predicted rating to [0, 1] range (where 1.0 is min rating and 5.0 is max rating)
            collab_score = (predicted_rating - 1.0) / 4.0
            
            # Combine scores
            combined_score = (content_weight * similarity) + ((1.0 - content_weight) * collab_score)
            
            hybrid_scores.append({
                'track_id': track_id,
                'track_name': row['track_name'],
                'artist_name': row['artist_name'],
                'genre': row['genre'],
                'popularity': row['popularity'],
                'similarity_score': round(similarity, 3),
                'predicted_rating': predicted_rating,
                'hybrid_score': round(combined_score, 3)
            })
            
        # Convert to DataFrame, sort, and return top_n
        df_hybrid = pd.DataFrame(hybrid_scores)
        df_hybrid = df_hybrid.sort_values(by='hybrid_score', ascending=False).head(top_n)
        
        return df_hybrid
