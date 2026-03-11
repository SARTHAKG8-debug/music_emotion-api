# Music Emotion Prediction Pipeline

This project predicts the emotional "mood" of a song using its audio features from Spotify. 
A Random Forest Classifier maps these features into four emotional quadrants:
- Happy/Joyful
- Sad/Depressed
- Calm/Relaxed
- Tense/Angry

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

The dataset `Music Info.csv` and `User Listening History.csv` should be present in the root folder alongside the `src` folder.
To train the random forest classifier:

```bash
cd src
python train_model.py
```

This will automatically clean the dataset, extract the required features, train the Random Forest on 80% of the dataset, evaluate its accuracy on the 20% test-set, and lastly save the `model.pkl` and `scaler.pkl` to the `api` directory.

### 3. Run the FastAPI Application

You can start the backend API server from the root of the application using `uvicorn`:

```bash
uvicorn api.main:app --reload
```

This will start a local server at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to see the interactive API documentation containing two endpoints.

#### `/predict` Endpoint
Allows you to POST arbitrary audio feature inputs (like danceability, tempo, etc.) to get an emotion prediction immediately.

#### `/predict_spotify` Endpoint
Given a Spotify Track ID (e.g. `09ZQ5TmUG8TSL56n0knqrj`), the pipeline fetches live feature data from the Spotify API and predicts its real-time emotion.

> **Note**: To use the `/predict_spotify` endpoint, you must create a Spotify Developer Application at [developer.spotify.com](https://developer.spotify.com/dashboard) and expose your credentials as environment variables:
> 
> For **Windows**:
> ```powershell
> $env:SPOTIPY_CLIENT_ID="your_client_id_here"
> $env:SPOTIPY_CLIENT_SECRET="your_client_secret_here"
> ```
> 
> For **Mac/Linux**:
> ```bash
> export SPOTIPY_CLIENT_ID="your_client_id_here"
> export SPOTIPY_CLIENT_SECRET="your_client_secret_here"
> ```
