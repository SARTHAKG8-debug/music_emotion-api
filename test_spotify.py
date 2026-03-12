import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

def test_spotify():
    # Use credentials from secrets.toml if possible, but for this test I'll hardcode or use env
    client_id = "cf1204eecab744e1a915be10b393c2ec"
    client_secret = "cea20210e0824b679302e9605534da01"
    
    print(f"Testing with Client ID: {client_id[:5]}...")
    
    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        track_id = '5wqMM6WOwXmX4rc1C3lUkd'
        
        print(f"--- Testing sp.track('{track_id}') ---")
        track = sp.track(track_id)
        print(f"Success! Track Name: {track['name']} by {track['artists'][0]['name']}")
        
        print(f"\n--- Testing sp.audio_features('{track_id}') ---")
        try:
            features = sp.audio_features(track_id)
            print(f"Success! Features: {features}")
        except Exception as e:
            print(f"Failed sp.audio_features: {e}")
            
    except Exception as e:
        print(f"Main Test Failed: {e}")

if __name__ == "__main__":
    test_spotify()
