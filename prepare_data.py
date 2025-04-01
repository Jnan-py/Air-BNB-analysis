import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

DATA_FOLDER = "data"
PROCESSED_FOLDER = "processed_data"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

features = ["neighbourhood", "latitude", "longitude", "room_type", "accommodates", 
            "bedrooms", "bathrooms", "beds", "minimum_nights", "availability_365"]
target = "price"

city_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

for file in city_files:
    city_name = file.replace(".csv", "")  
    print(f"📌 Processing {city_name} dataset...")
    
    df = pd.read_csv(os.path.join(DATA_FOLDER, file))
    
    df["City"] = city_name
    
    available_features = [col for col in features if col in df.columns]
    required_columns = available_features + [target] if target in df.columns else available_features
    df = df[required_columns]
    
    categorical_features = ["room_type", "neighbourhood"]
    for col in categorical_features:
        if col in df.columns:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
    
    processed_file = os.path.join(PROCESSED_FOLDER, f"{city_name}_processed.csv")
    df.to_csv(processed_file, index=False)
    print(f"✅ Processed {city_name} dataset saved: {processed_file}")

print("🎉 All datasets processed successfully!")
