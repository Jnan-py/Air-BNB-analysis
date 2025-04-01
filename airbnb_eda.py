import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "data"

cities = ["New_York", "London", "Paris", "Berlin", "Tokyo", "Sydney", 
          "Toronto", "Los_Angeles", "Barcelona", "Amsterdam"]

os.makedirs("plots", exist_ok=True)

for city in cities:
    file_path = os.path.join(data_dir, f"{city}.csv")

    if os.path.exists(file_path):
        print(f"📊 Analyzing {city}...")

        df = pd.read_csv(file_path)
        
        print(f"\n📌 Summary statistics for {city}:")
        print(df.describe())

        plt.figure(figsize=(8, 5))
        sns.histplot(df["price"], bins=50, kde=True)
        plt.xlim(0, 500)  
        plt.title(f"Price Distribution in {city}")
        plt.xlabel("Price ($)")
        plt.ylabel("Count")
        plt.savefig(f"plots/{city}_price_distribution.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.countplot(y=df["room_type"], order=df["room_type"].value_counts().index)
        plt.title(f"Room Type Distribution in {city}")
        plt.xlabel("Count")
        plt.ylabel("Room Type")
        plt.savefig(f"plots/{city}_room_type_distribution.png")
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.histplot(df["availability_365"], bins=30, kde=True)
        plt.title(f"Availability of Listings in {city}")
        plt.xlabel("Days Available per Year")
        plt.ylabel("Count")
        plt.savefig(f"plots/{city}_availability.png")
        plt.close()

        print(f"✅ EDA completed for {city}. Plots saved in 'plots/' folder.\n")

    else:
        print(f"❌ File not found: {file_path}")

print("🎉 EDA completed for all cities!")
