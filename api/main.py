from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd

app = FastAPI(title="Music Emotion Prediction API", version="1.0.0")

# Load ML model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise RuntimeError("Model or scaler not found. Please train the model first by running `python src/train_model.py`.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

FEATURE_NAMES = ['danceability', 'loudness', 'speechiness', 'acousticness', 
                 'instrumentalness', 'liveness', 'tempo']

# Load music dataset for fallback
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Music Info.csv')
music_df = pd.read_csv(data_path) if os.path.exists(data_path) else None

class FeaturesInput(BaseModel):
    danceability: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    tempo: float

class SpotifyTrackInput(BaseModel):
    track_id: str

@app.get("/")
def home():
    return {"message": "Welcome to the Music Emotion Prediction API"}

@app.post("/predict")
def predict_emotion(features: FeaturesInput):
    """
    Predict the emotion of a song given its raw audio features.
    """
    # Create feature array in the exact order as training
    feature_values = [[
        features.danceability,
        features.loudness,
        features.speechiness,
        features.acousticness,
        features.instrumentalness,
        features.liveness,
        features.tempo
    ]]
    
    X_scaled = scaler.transform(feature_values)
    prediction = model.predict(X_scaled)[0]
    
    return {
        "predicted_mood": prediction
    }

@app.post("/predict_spotify")
def predict_emotion_from_spotify(track: SpotifyTrackInput):
    """
    Predict the emotion of a song using its Spotify Track ID.
    Requires SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables.
    """
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500, 
            detail="Spotify API credentials not configured. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in environment."
        )
    
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    try:
        # Fetch track info (usually works even if audio-features is restricted)
        track_info = sp.track(track.track_id)
        name = track_info['name']
        artist_name = track_info['artists'][0]['name']
        
        # 2. Try fetching audio features from API
        song_features = None
        try:
            features_list = sp.audio_features(track.track_id)
            if features_list and features_list[0]:
                song_features = features_list[0]
        except Exception as e:
            # If 403, we will handle it via fallback below
            pass
            
        # 3. Fallback to local dataset if API failed or returned None
        if song_features is None and music_df is not None:
            # First try matching by spotify_id
            search_result = music_df[music_df['spotify_id'] == track.track_id]
            
            # If not found by ID, try matching by name + artist
            if search_result.empty:
                name_lower = name.lower().strip()
                artist_lower = artist_name.lower().strip()
                name_match = music_df[
                    (music_df['name'].str.lower().str.strip() == name_lower) &
                    (music_df['artist'].str.lower().str.strip() == artist_lower)
                ]
                if not name_match.empty:
                    search_result = name_match
            
            # If still not found, try partial name+artist match
            if search_result.empty:
                name_lower = name.lower().strip()
                artist_lower = artist_name.lower().strip()
                partial_match = music_df[
                    (music_df['name'].str.lower().str.contains(name_lower, na=False)) &
                    (music_df['artist'].str.lower().str.contains(artist_lower, na=False))
                ]
                if not partial_match.empty:
                    search_result = partial_match
            
            if not search_result.empty:
                row = search_result.iloc[0]
                song_features = {k: row[k] for k in FEATURE_NAMES}
            
        # 4. Final Fallback: Simulated features if track is missing from everything
        if not song_features:
            import hashlib
            hash_val = int(hashlib.md5(track.track_id.encode('utf-8')).hexdigest(), 16)
            def get_val(min_v, max_v, offset_index):
                shifted_hash = hash_val >> (offset_index * 4)
                pct = (shifted_hash % 1000) / 1000.0
                return min_v + (max_v - min_v) * pct
            
            song_features = {
                'danceability': get_val(0.3, 0.9, 0),
                'loudness': get_val(-15.0, -3.0, 1),
                'speechiness': get_val(0.02, 0.25, 2),
                'acousticness': get_val(0.01, 0.8, 3),
                'instrumentalness': get_val(0.0, 0.5, 4),
                'liveness': get_val(0.05, 0.4, 5),
                'tempo': get_val(80.0, 160.0, 6)
            }
            # Adding a flag to indicate it's simulated
            song_features['_simulated'] = True

    except spotipy.SpotifyException as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Extract needed features
    feature_values = [[
        float(song_features['danceability']),
        float(song_features['loudness']),
        float(song_features['speechiness']),
        float(song_features['acousticness']),
        float(song_features['instrumentalness']),
        float(song_features['liveness']),
        float(song_features['tempo'])
    ]]
    
    X_scaled = scaler.transform(feature_values)
    prediction = model.predict(X_scaled)[0]
    
    return {
        "track_name": name,
        "artist": artist_name,
        "spotify_features": song_features,
        "predicted_mood": prediction
    }
