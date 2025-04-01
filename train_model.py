import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import *
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_FOLDER = "processed_data"
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

processed_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith("_processed.csv")]

for file in processed_files:
    city_name = file.replace("_processed.csv", "")
    print(f"📌 Training model for {city_name}...")
    
    df = pd.read_csv(os.path.join(PROCESSED_FOLDER, file))

    target = "price"
    if target not in df.columns:
        print(f"❌ Skipping {city_name}: No 'price' column found!")
        continue

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])  
    y = df[target]  

    categorical_cols = X.select_dtypes(include=['object']).columns

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(X[categorical_cols])

    encoder_path = os.path.join(MODEL_FOLDER, f"{city_name}_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    print(f"📂 Encoder saved: {encoder_path}")

    X = X.drop(columns=categorical_cols)
    X = np.hstack((X.values, encoded_features))

    feature_names = list(range(X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation ='relu'),    
        Dense(64,activation ='relu'),        
        Dense(32,activation ='relu'),        
        Dense(1)  

    ])
    
    model.compile(optimizer='adam', loss='mse')    

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"✅ {city_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, r2 : {r2:.2f}")

    model_data = {
        'model': model,
        'features': feature_names
    }

    model_path = os.path.join(MODEL_FOLDER, f"{city_name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"📂 Model saved: {model_path}\n")

print("🎉 All city models trained and saved!")

    