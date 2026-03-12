import streamlit as st
import joblib
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np

# Feature mapping (same order as used in training) - defined early so load_data() can use it
FEATURE_NAMES = ['danceability', 'loudness', 'speechiness', 'acousticness', 
                 'instrumentalness', 'liveness', 'tempo']

# Set up page configurations
st.set_page_config(page_title="Music Emotion Predictor", page_icon="🎵", layout="centered")

# Custom CSS for aesthetic, premium dark mode styling
st.markdown("""
<style>
/* Style the main title */
h1 {
    color: #1DB954;
    text-align: center;
    font-weight: 800;
    margin-bottom: 0px;
}
/* Subtitle */
.subtitle {
    text-align: center;
    color: #a0a0a0;
    font-size: 1.1rem;
    margin-bottom: 30px;
}
/* Prediction Box Styles */
.mood-box {
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease;
}
.mood-box:hover {
    transform: translateY(-5px);
}
.mood-title {
    font-size: 1.2rem;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.mood-result {
    font-size: 2.5rem;
    font-weight: 800;
}
/* Individual Mood Colors */
.happy { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); color: #333; }
.sad { background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); color: #333; }
.calm { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: #333; }
.tense { background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%); color: #fff; }
.default-mood { background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%); color: #fff; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # Model files are stored in the `api` folder relative to this script
    model_path = os.path.join(os.path.dirname(__file__), 'api', 'model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'api', 'scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

def get_mood_html(prediction):
    # Map the prediction string to a class for styling
    pred_lower = str(prediction).lower()
    if 'happy' in pred_lower or 'joy' in pred_lower:
        css_class = 'happy'
        emoji = '☀️'
    elif 'sad' in pred_lower or 'depress' in pred_lower:
        css_class = 'sad'
        emoji = '🌧️'
    elif 'calm' in pred_lower or 'relax' in pred_lower:
        css_class = 'calm'
        emoji = '🍃'
    elif 'tense' in pred_lower or 'angry' in pred_lower:
        css_class = 'tense'
        emoji = '🔥'
    else:
        css_class = 'default-mood'
        emoji = '🎶'
        
    return f'''
    <div class="mood-box {css_class}">
        <div class="mood-title">Predicted Mood</div>
        <div class="mood-result">{emoji} {prediction}</div>
    </div>
    '''

@st.cache_data
def load_data():
    # Load the music dataset to use as a fallback for audio features
    data_path = os.path.join(os.path.dirname(__file__), 'Music Info.csv')
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        # Keep only necessary columns to save memory
        cols_to_keep = ['spotify_id', 'name', 'artist'] + FEATURE_NAMES
        return df[cols_to_keep]
    return None

# Main App Execution
st.title("🎵 Music Emotion Predictor")
st.markdown("<div class='subtitle'>Predict the emotional mood of a song using its Spotify audio features.</div>", unsafe_allow_html=True)

model, scaler = load_models()
music_df = load_data()

if model is None or scaler is None:
    st.error("⚠️ **Model files not found.** Please ensure `api/model.pkl` and `api/scaler.pkl` exist by running the training script locally before deploying.")
    st.stop()

# FEATURE_NAMES is defined at the top of the file

tab1, tab2 = st.tabs(["🎧 Predict from Spotify", "🎛️ Manual Audio Features"])

with tab1:
    st.markdown("### Enter a Spotify Track ID")
    st.write("E.g., `0keNu0t0tqsWtExGM3nT1D` (Mr. Brightside) or `09ZQ5TmUG8TSL56n0knqrj` (In the End)")
    track_input = st.text_input("Track ID or Link", "")
    
    # Try to extract track ID if they pasted a link
    if "open.spotify.com/track/" in track_input:
        track_id = track_input.split("open.spotify.com/track/")[1].split("?")[0]
    else:
        track_id = track_input.strip()
        
    if st.button("Predict 🔮", key="spotify_predict"):
        if not track_id:
            st.warning("Please enter a valid Track ID.")
        else:
            client_id = os.environ.get("SPOTIPY_CLIENT_ID")
            client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
            
            # Try getting from st.secrets if not in env (for Streamlit Cloud deployment)
            if not client_id or not client_secret:
                try:
                    client_id = client_id or st.secrets.get("SPOTIPY_CLIENT_ID")
                    client_secret = client_secret or st.secrets.get("SPOTIPY_CLIENT_SECRET")
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
            
            if not client_id or not client_secret:
                st.error("Spotify API credentials are not configured. Please set `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET` in your environment variables or Streamlit Secrets.")
            else:
                try:
                    with st.spinner("Fetching track details from Spotify..."):
                        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
                        sp = spotipy.Spotify(auth_manager=auth_manager)
                        
                        # 1. Get track info (usually works even if audio-features is restricted)
                        track_info = sp.track(track_id)
                        name = track_info['name']
                        artist_name = track_info['artists'][0]['name']
                        album_art_url = track_info['album']['images'][0]['url'] if track_info['album']['images'] else None
                        
                        # 2. Try fetching audio features from API
                        song_features = None
                        try:
                            features_list = sp.audio_features(track_id)
                            if features_list and features_list[0]:
                                song_features = features_list[0]
                        except Exception as api_err:
                            if "403" in str(api_err):
                                st.info("ℹ️ **Spotify API restriction detected.** Falling back to local dataset for audio features...")
                            else:
                                raise api_err
                        
                        # 3. Fallback to local dataset if API failed
                        if song_features is None and music_df is not None:
                            # First try matching by spotify_id
                            search_result = music_df[music_df['spotify_id'] == track_id]
                            
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
                                    st.info(f"✅ Found **\"{name}\"** by **{artist_name}** in our database via name matching!")
                            
                            # If still not found, try partial name match (artist only)
                            if search_result.empty:
                                artist_lower = artist_name.lower().strip()
                                name_lower = name.lower().strip()
                                partial_match = music_df[
                                    (music_df['name'].str.lower().str.contains(name_lower, na=False)) &
                                    (music_df['artist'].str.lower().str.contains(artist_lower, na=False))
                                ]
                                if not partial_match.empty:
                                    search_result = partial_match
                                    matched_name = search_result.iloc[0]['name']
                                    matched_artist = search_result.iloc[0]['artist']
                                    st.info(f"✅ Found a close match: **\"{matched_name}\"** by **{matched_artist}** in our database!")
                            
                            if not search_result.empty:
                                row = search_result.iloc[0]
                                song_features = {k: row[k] for k in FEATURE_NAMES}
                            
                        # 4. Final Fallback: Simulated features for fun prediction
                        if song_features is None:
                            st.info(f"ℹ️ **\"{name}\" by {artist_name} not found in our database.** Since Spotify resticted the 'audio-features' API (403), we are using **simulated audio features** to give you a fun prediction anyway!")
                            import hashlib
                            # Deterministically generate features from track_id
                            hash_val = int(hashlib.md5(track_id.encode('utf-8')).hexdigest(), 16)
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

                        if song_features:
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
                            
                            st.divider()
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if album_art_url:
                                    st.image(album_art_url, use_container_width=True)
                            with col2:
                                st.subheader(f"{name}")
                                st.write(f"by **{artist_name}**")
                                st.markdown(get_mood_html(prediction), unsafe_allow_html=True)
                                
                            with st.expander("View Audio Features"):
                                st.json({k: song_features[k] for k in FEATURE_NAMES})
                                
                except Exception as e:
                    if "404" in str(e):
                        st.error("Track not found on Spotify. Please check the ID.")
                    else:
                        st.error(f"Error: {e}")

with tab2:
    st.markdown("### Tweak Audio Features manually")
    
    col1, col2 = st.columns(2)
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0)
    with col2:
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.1)

    if st.button("Predict Emotion 🔮", key="manual_predict"):
        feature_values = [[
            danceability,
            loudness,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            tempo
        ]]
        
        X_scaled = scaler.transform(feature_values)
        prediction = model.predict(X_scaled)[0]
        st.markdown(get_mood_html(prediction), unsafe_allow_html=True)
