import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys

# Add src to system path to import recommendation models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from recommender import ContentRecommender, CollaborativeRecommender, HybridRecommender

# Set page config with Spotify-like logo and layout
st.set_page_config(
    page_title="EchoVibe - Premium Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Spotify Dark Aesthetic
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header */
    .header-container {
        background: linear-gradient(135deg, #1DB954 0%, #0b3d1c 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #b3b3b3;
        margin-top: 0.5rem;
    }
    
    /* Card design for metrics and results */
    .track-card {
        background-color: #181818;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1DB954;
        margin-bottom: 1rem;
        transition: transform 0.2s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .track-card:hover {
        transform: translateY(-2px);
        background-color: #282828;
    }
    .track-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .track-artist {
        font-size: 1rem;
        color: #1DB954;
        font-weight: 500;
    }
    .track-meta {
        font-size: 0.85rem;
        color: #b3b3b3;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #282828;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #b3b3b3;
        font-size: 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #1DB954 !important;
        border-bottom-color: #1DB954 !important;
    }
    
    /* Custom Slider & Selectors color */
    .stSlider [data-testid="stTickBar"] {
        color: #b3b3b3;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading so the app starts instantly after the first load
@st.cache_data
def load_data():
    # DATA_DIR = r"C:\Users\HP\Documents\KAIF\Slash\TASK 6\data"
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    SPOTIFY_DATA_PATH = os.path.join(DATA_DIR, "SpotifyFeatures.csv")
    USER_DATA_PATH = os.path.join(DATA_DIR, "user_interactions.csv")
    
    df_tracks = pd.read_csv(SPOTIFY_DATA_PATH)
    df_interactions = pd.read_csv(USER_DATA_PATH)
    return df_tracks, df_interactions

@st.cache_resource
def load_recommenders(_df_tracks, _df_interactions):
    content_rec = ContentRecommender(_df_tracks)
    collab_rec = CollaborativeRecommender(_df_interactions, _df_tracks)
    hybrid_rec = HybridRecommender(content_rec, collab_rec)
    return content_rec, collab_rec, hybrid_rec

# App initialization
try:
    df_tracks, df_interactions = load_data()
    content_rec, collab_rec, hybrid_rec = load_recommenders(df_tracks, df_interactions)
except Exception as e:
    st.error(f"Error loading datasets. Please make sure data preprocessing completed. Details: {e}")
    st.stop()

# Header Section
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🎵 EchoVibe</h1>
    <div class="header-subtitle">Advanced Music Recommendation System powered by Collaborative, Content-Based, & Hybrid Filtering</div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Settings & Statistics
with st.sidebar:
    st.markdown("<h2 style='color: #1DB954;'>📊 Library Stats</h2>", unsafe_allow_html=True)
    st.metric("Total Tracks", f"{len(df_tracks):,}")
    st.metric("Unique Artists", f"{df_tracks['artist_name'].nunique():,}")
    st.metric("Genres Available", f"{df_tracks['genre'].nunique():,}")
    st.metric("Simulated Active Users", f"{df_interactions['user_id'].nunique():,}")
    
    st.markdown("<hr style='border-color: #282828;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1DB954;'>⚙️ Preferences</h3>", unsafe_allow_html=True)
    rec_count = st.slider("Recommendations Count", 5, 20, 10)
    
    st.markdown("<br><br><span style='color: #555555; font-size: 0.8rem;'>Slash Mark Internship - Task 6</span>", unsafe_allow_html=True)

# Create Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Similarity Match (Content)", 
    "👤 Personalized for You (Collaborative)", 
    "⚡ Intelligent Mix (Hybrid)"
])

# Audio features columns for visualizations
AUDIO_FEATS = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']

def plot_audio_features(song_details, title="Audio Fingerprint"):
    """Plot radar chart for a song's audio features."""
    features_val = [song_details[f].values[0] for f in AUDIO_FEATS]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=features_val + [features_val[0]],
        theta=AUDIO_FEATS + [AUDIO_FEATS[0]],
        fill='toself',
        fillcolor='rgba(29, 185, 84, 0.3)',
        line=dict(color='#1DB954', width=2),
        name=song_details['track_name'].values[0]
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='#181818'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        title=dict(text=title, font=dict(size=16, color='#FFFFFF')),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# ----------------- TAB 1: CONTENT-BASED RECOMMENDATIONS -----------------
with tab1:
    st.subheader("Discover Songs with Similar Audio Attributes")
    st.markdown("Enter a track name to find other tracks that share its acoustic profile, energy, tempo, and genre.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        search_query = st.text_input("Type Track Name to Search", "Don't Let Me Be Lonely Tonight", key="c_search")
        
        # Filter matching tracks to let user select
        matching_tracks = df_tracks[df_tracks['track_name'].str.lower().str.contains(search_query.lower(), na=False)]
        
        if not matching_tracks.empty:
            # Sort by popularity to present best matches first
            matching_tracks = matching_tracks.sort_values(by='popularity', ascending=False).head(10)
            track_options = [f"{row['track_name']} - {row['artist_name']} ({row['genre']})" for idx, row in matching_tracks.iterrows()]
            
            selected_track_str = st.selectbox("Select Exact Song from Matches", track_options, key="c_select")
            
            # Extract name and artist
            selected_idx = track_options.index(selected_track_str)
            selected_row = matching_tracks.iloc[selected_idx]
            selected_track_name = selected_row['track_name']
            selected_artist_name = selected_row['artist_name']
            
            # Show details of selected track
            st.markdown("---")
            st.markdown(f"#### Selected Track Signature:")
            st.markdown(f"**Title**: {selected_track_name}")
            st.markdown(f"**Artist**: {selected_artist_name}")
            st.markdown(f"**Genre**: {selected_row['genre']} | **Popularity**: {selected_row['popularity']}/100")
            st.markdown(f"**Tempo**: {selected_row['tempo']:.0f} BPM | **Key**: {selected_row['key']}")
            
            # Plot features
            song_df = df_tracks[(df_tracks['track_name'] == selected_track_name) & (df_tracks['artist_name'] == selected_artist_name)].head(15)
            fig = plot_audio_features(song_df, f"Acoustic DNA: {selected_track_name}")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No matches found. Try another search.")
            selected_track_name = None

    with col2:
        if selected_track_name:
            st.markdown(f"### 🎯 Top {rec_count} Recommendations Similar to **{selected_track_name}**")
            
            with st.spinner("Finding matches..."):
                recs = content_rec.get_recommendations(selected_track_name, selected_artist_name, top_n=rec_count)
                
            if not recs.empty:
                for idx, row in recs.iterrows():
                    st.markdown(f"""
                    <div class="track-card">
                        <div class="track-title">{row['track_name']}</div>
                        <div class="track-artist">By {row['artist_name']}</div>
                        <div class="track-meta">
                            <b>Genre</b>: {row['genre']} | 
                            <b>Popularity</b>: {row['popularity']} | 
                            <b>Match Confidence</b>: <span style='color: #1DB954; font-weight: bold;'>{row['similarity_score']*100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recommendations found.")


# ----------------- TAB 2: COLLABORATIVE RECOMMENDATIONS -----------------
with tab2:
    st.subheader("Personalized Listening History Recommendations")
    st.markdown("Select a simulated user to view their listening history and get recommendations based on similar user preferences.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # User dropdown
        user_list = sorted(df_interactions['user_id'].unique())
        selected_user = st.selectbox("Select Active User", user_list, key="u_select")
        
        # Display User's listening profile
        user_history = df_interactions[df_interactions['user_id'] == selected_user].merge(df_tracks, on='track_id')
        user_history = user_history.sort_values(by='rating', ascending=False)
        
        st.markdown(f"#### {selected_user}'s Top Tracks:")
        for idx, row in user_history.head(5).iterrows():
            st.markdown(f"- **{row['track_name']}** by *{row['artist_name']}* (Rated: {row['rating']} ⭐, Listened: {row['listen_count']} times)")
            
        # Draw genre preference bar chart
        st.markdown("---")
        st.markdown("#### Favorite Genres")
        genre_counts = user_history['genre'].value_counts()
        fig_genre = px.bar(
            x=genre_counts.index[:5], 
            y=genre_counts.values[:5],
            labels={'x': 'Genre', 'y': 'Count'},
            color_discrete_sequence=['#1DB954']
        )
        fig_genre.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            height=250,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_genre, use_container_width=True)

    with col2:
        st.markdown(f"### 👥 Personalized Recommendations for **{selected_user}**")
        
        with st.spinner("Analyzing profile..."):
            recs_collab = collab_rec.get_recommendations(selected_user, top_n=rec_count)
            
        if not recs_collab.empty:
            for idx, row in recs_collab.iterrows():
                st.markdown(f"""
                <div class="track-card">
                    <div class="track-title">{row['track_name']}</div>
                    <div class="track-artist">By {row['artist_name']}</div>
                    <div class="track-meta">
                        <b>Genre</b>: {row['genre']} | 
                        <b>Popularity</b>: {row['popularity']} | 
                        <b>Predicted Rating</b>: <span style='color: #1DB954; font-weight: bold;'>{row['predicted_rating']:.1f} / 5.0 ⭐</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recommendations found.")


# ----------------- TAB 3: HYBRID RECOMMENDATIONS -----------------
with tab3:
    st.subheader("Hybrid Recommender System")
    st.markdown("The hybrid recommender combines content similarity to a last played song with your collaborative filtering profile.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_user_h = st.selectbox("Select User Profile", user_list, key="h_user")
        
        # Search last played song
        search_query_h = st.text_input("Search Last Listened Track", "Someone Like You", key="h_search")
        matching_tracks_h = df_tracks[df_tracks['track_name'].str.lower().str.contains(search_query_h.lower(), na=False)]
        
        if not matching_tracks_h.empty:
            matching_tracks_h = matching_tracks_h.sort_values(by='popularity', ascending=False).head(10)
            track_options_h = [f"{row['track_name']} - {row['artist_name']} ({row['genre']})" for idx, row in matching_tracks_h.iterrows()]
            selected_track_str_h = st.selectbox("Select Last Track", track_options_h, key="h_select")
            
            selected_idx_h = track_options_h.index(selected_track_str_h)
            selected_row_h = matching_tracks_h.iloc[selected_idx_h]
            selected_track_name_h = selected_row_h['track_name']
            selected_artist_name_h = selected_row_h['artist_name']
        else:
            st.warning("No matches found.")
            selected_track_name_h = None
            
        st.markdown("---")
        st.markdown("#### ⚖️ Recommendation Blend")
        content_w = st.slider("Content Weight (Acoustic Match)", 0.0, 1.0, 0.5, 0.05)
        collab_w = 1.0 - content_w
        st.markdown(f"- **Acoustic profile weight**: {content_w*100:.0f}%\n- **User taste weight**: {collab_w*100:.0f}%")

    with col2:
        if selected_track_name_h:
            st.markdown(f"### ⚡ Hybrid Recommendations for **{selected_user_h}** (Based on **{selected_track_name_h}**)")
            
            with st.spinner("Computing hybrid scoring..."):
                recs_hybrid = hybrid_rec.get_recommendations(
                    selected_user_h, 
                    selected_track_name_h, 
                    selected_artist_name_h, 
                    top_n=rec_count, 
                    content_weight=content_w
                )
                
            if not recs_hybrid.empty:
                for idx, row in recs_hybrid.iterrows():
                    st.markdown(f"""
                    <div class="track-card">
                        <div class="track-title">{row['track_name']}</div>
                        <div class="track-artist">By {row['artist_name']}</div>
                        <div class="track-meta">
                            <b>Genre</b>: {row['genre']} | 
                            <b>Similarity</b>: {row['similarity_score']*100:.0f}% | 
                            <b>Collab rating</b>: {row['predicted_rating']:.1f}⭐ | 
                            <b>Combined Score</b>: <span style='color: #1DB954; font-weight: bold;'>{row['hybrid_score']:.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recommendations found.")
