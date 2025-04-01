import pandas as pd
import os

cities = {
    "New_York": "New_York_listings.csv",
    "London": "London_listings.csv",
    "Paris": "Paris_listings.csv",
    "Berlin": "Berlin_listings.csv",
    "Tokyo": "Tokyo_listings.csv",
    "Sydney": "Sydney_listings.csv",
    "Toronto": "Toronto_listings.csv",
    "Los_Angeles": "Los_Angeles_listings.csv",
    "Barcelona": "Barcelona_listings.csv",
    "Amsterdam": "Amsterdam_listings.csv"
}

raw_data_dir = "raw_data"
clean_data_dir = "data"

os.makedirs(clean_data_dir, exist_ok=True)

columns_to_keep = [
    "id", "name", "host_id", "host_name", "neighbourhood",
    "latitude", "longitude", "room_type", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month", "availability_365"
]

for city, filename in cities.items():
    file_path = os.path.join(raw_data_dir, filename)
    
    if os.path.exists(file_path):
        print(f"Processing {city} data...")
        
        df = pd.read_csv(file_path, low_memory=False)
        
        df["City"] = city
        
        df_cleaned = df[columns_to_keep + ["City"]]
        
        output_path = os.path.join(clean_data_dir, f"{city}.csv")
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"✅ {city} dataset saved at {output_path}")
    else:
        print(f"❌ File not found: {file_path}")

print("🎉 All datasets processed successfully!")
