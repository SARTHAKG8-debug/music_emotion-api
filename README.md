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

### 4. Run the Streamlit Frontend

A Streamlit web application is provided for a more interactive, UI-driven experience rather than just raw API endpoints.

To run the Streamlit frontend locally:

```bash
streamlit run streamlit_app.py
```

This will open a browser window at `http://localhost:8501` where you can predict emotion by either pasting a Spotify track ID, or by manually adjusting music feature sliders.

### 5. Deploying the Streamlit App

The easiest way to make this app accessible on the internet is via [Streamlit Community Cloud](https://share.streamlit.io):

1. **Push your code to GitHub**, making sure `streamlit_app.py`, `requirements.txt`, and the `api/` folder (with `.pkl` models) are included.
2. Log into Streamlit Community Cloud and click **New app**.
3. Select this repository and branch.
4. Set the **Main file path** to `streamlit_app.py`.
5. Before clicking "Deploy!", click **Advanced settings** and add your Spotify credentials in the **Secrets** section:
   ```toml
   SPOTIPY_CLIENT_ID="your_client_id_here"
   SPOTIPY_CLIENT_SECRET="your_client_secret_here"
   ```
6. Click **Deploy!**
