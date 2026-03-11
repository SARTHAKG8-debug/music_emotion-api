from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np

app = FastAPI(title="Music Emotion Prediction API", version="1.0.0")

# Load ML model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise RuntimeError("Model or scaler not found. Please train the model first by running `python src/train_model.py`.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Required features in specific order used for training
FEATURE_NAMES = ['danceability', 'loudness', 'speechiness', 'acousticness', 
                 'instrumentalness', 'liveness', 'tempo']

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
        # Fetch track info to get the name and artist
        track_info = sp.track(track.track_id)
        name = track_info['name']
        artist_name = track_info['artists'][0]['name']
        
        # Fetch audio features
        features_list = sp.audio_features(track.track_id)
        if not features_list or features_list[0] is None:
            raise HTTPException(status_code=404, detail="Could not find audio features for this track.")
            
        song_features = features_list[0]
        
    except spotipy.SpotifyException as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Extract needed features
    feature_values = [[
        song_features['danceability'],
        song_features['loudness'],
        song_features['speechiness'],
        song_features['acousticness'],
        song_features['instrumentalness'],
        song_features['liveness'],
        song_features['tempo']
    ]]
    
    X_scaled = scaler.transform(feature_values)
    prediction = model.predict(X_scaled)[0]
    
    return {
        "track_name": name,
        "artist": artist_name,
        "spotify_features": song_features,
        "predicted_mood": prediction
    }
