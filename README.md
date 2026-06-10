# Build a Music Recommendation System

This project is an advanced-level machine learning implementation of a **Music Recommendation System**. It integrates content-based filtering on song audio characteristics with collaborative filtering on user interaction logs to build a high-performance, robust hybrid recommendation engine.

The project features a fully interactive, Spotify-themed **Streamlit Web Application** for exploring recommendations, analyzing song audio signatures, and customizing recommendation parameters in real-time.

---

## 📂 Project Structure
```text
TASK /
├── app/
│   └── app.py                 # Spotify-themed interactive Streamlit Web App
├── data/
│   ├── SpotifyFeatures.csv    # Raw track metadata and audio features (~32MB)
│   └── user_interactions.csv  # User interaction logs (ratings, listening counts)
├── notebooks/
│   └── Music_Recommendation_System.ipynb # In-depth EDA & model explanations
├── src/
│   ├── data_preprocessing.py  # Script to download datasets & generate user history
│   ├── recommender.py         # Recommendation algorithms (Content, Collab, Hybrid)
│   └── verify.py              # Validation script to test the models
├── requirements.txt           # Project dependencies
└── README.md                  # This documentation file
```

---

## ⚡ Recommendation Engine Overview

### 1. Content-Based Filtering (`ContentRecommender`)
- **Acoustic Signature Match**: Recommends songs similar to a seed track using **Cosine Similarity** on normalized audio features (acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, popularity).
- **Genre Matching weight**: One-hot encodes the track's genre and applies a `1.5x` weight multiplier to give matching genres higher matching priority.
- **On-the-Fly computation**: Compares similarity vectors on-the-fly for real-time memory efficiency.

### 2. Collaborative Filtering (`CollaborativeRecommender`)
- **User-Preferences Match**: Recommends songs to an active user based on user profiles with similar music tastes.
- **Matrix Factorization (SVD)**: Constructs a sparse user-item matrix of rating residuals (centered around user means) and applies **Singular Value Decomposition (SVD)** to capture latent factors.
- **Cold-Start Fallback**: Recommends top globally popular tracks when a new user is queried.

### 3. Hybrid Recommender (`HybridRecommender`)
- Combines content similarity and collaborative ratings using a weighted ensemble score:
  $$\text{Hybrid Score} = w \cdot \text{Similarity Score} + (1 - w) \cdot \text{Normalized Predicted Rating}$$
- Sliders in the Streamlit app allow dynamically shifting the weight between the two models.

---

## 🚀 Getting Started & How to Run

### Prerequisite: Set up your environment
1. Open your terminal or Command Prompt.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Option 1: Run the Interactive Web Dashboard (Recommended)
Launch the Spotify-themed Streamlit application:
```bash
streamlit run app/app.py
```
This will open the application in your default web browser (typically at `http://localhost:8501`). In this app, you can:
- Type in song titles to search and analyze their audio fingerprints (visualized via dynamic radar charts).
- Retrieve similarity recommendations, user preference recommendations, or a custom hybrid blend.
- Adjust sliders to change weights or recommendations count on-the-fly.

### Option 2: Run step-by-step Jupyter Notebook
If you want to view the data analysis, code logic, and plotting code, open the notebook:
```bash
jupyter notebook notebooks/Music_Recommendation_System.ipynb
```
Or open it inside VS Code / Jupyter Lab.

### Option 3: Run the Command-Line Verification Script
To verify that all models are configured correctly and running without a browser GUI, execute:
```bash
python src/verify.py
```
This outputs test recommendations for a sample song and user profile.

```text
Built by

 __  __  ___  _   _    _    __  __ __  __ _____ ____  
|  \/  |/ _ \| | | |  / \  |  \/  |  \/  | ____|  _ \ 
| |\/| | | | | |_| | / _ \ | |\/| | |\/| |  _| | | | |
| |  | | |_| |  _  |/ ___ \| |  | | |  | | |___| |_| |
|_|  |_|\___/|_| |_/_/   \_\_|  |_|_|  |_|_____|____/

 _  __    _    ___ _____
| |/ /   / \  |_ _|  ___|
| ' /   / _ \  | || |_
| . \  / ___ \ | ||  _|
|_|\_\/_/   \_\___|_|
```
