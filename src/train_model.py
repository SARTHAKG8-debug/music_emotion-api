import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    print("Loading datasets...")
    # Load data
    df = pd.read_csv('../Music Info.csv')
    
    # Drop rows with missing values in our features of interest
    features = ['danceability', 'loudness', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'tempo']
    df = df.dropna(subset=features + ['valence', 'energy'])
    
    print(f"Dataset size after cleaning: {len(df)}")
    
    # Create the target variable "Mood" based on Russell's Circumplex Model
    # We use median of valence and energy to divide into 4 quadrants
    v_med = df['valence'].median()
    e_med = df['energy'].median()
    
    print(f"Valence median: {v_med:.3f}, Energy median: {e_med:.3f}")
    
    def get_mood(row):
        v = row['valence']
        e = row['energy']
        if v >= v_med and e >= e_med:
            return 'Happy/Joyful'
        elif v >= v_med and e < e_med:
            return 'Calm/Relaxed'
        elif v < v_med and e >= e_med:
            return 'Tense/Angry'
        else:
            return 'Sad/Depressed'
            
    df['mood'] = df.apply(get_mood, axis=1)
    
    X = df[features]
    y = df['mood']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    model_dir = '../api'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    main()
