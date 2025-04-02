# Airbnb Data Analysis & Recommendation System

This repository contains a Streamlit application that performs data cleaning, exploratory data analysis, predictive modeling, and recommendation generation for Airbnb listings across various cities.

## Overview

The **Airbnb Data Analysis & Recommendation System** provides:

1. **Data Cleaning & Preparation**: Converts raw Airbnb CSV files into cleaned and processed datasets.
2. **Exploratory Data Analysis**: Generates insights on price distribution, room type, and neighborhood statistics.
3. **Predictive Modeling**: Trains a model to predict the price of Airbnb listings given specific features.
4. **Recommendation System**: Recommends listings based on description similarity, geographic proximity, neighborhood, and room type.

---

## Project Structure

```

.
├── data
│ ├── raw_data
│ │ ├── Amsterdam_listings.csv
│ │ ├── Berlin_listings.csv
│ │ └── ...
│ └── processed_data
│ ├── Amsterdam_processed.csv
│ ├── Berlin_processed.csv
│ └── ...
├── models
│ ├── Amsterdam_model.pkl
│ ├── Amsterdam_encoder.pkl
│ └── ...
├── plots
├── clean_airbnb_data.py
├── prepare_data.py
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

```

- **`raw_data`**: Contains the original Airbnb listings CSV files.
- **`processed_data`**: Contains the cleaned/processed CSV files used by the app.
- **`models`**: Contains trained model `.pkl` files and corresponding encoders.
- **`app.py`**: Main Streamlit application.

---

## Data Flow

1. **`clean_airbnb_data.py`**

   - Reads the raw CSV files from `raw_data`.
   - Cleans and transforms them (removing nulls, unnecessary columns, etc.).
   - Outputs interim or cleaned files (optional).

2. **`prepare_data.py`**

   - Reads the cleaned files, performs additional feature engineering or processing.
   - Saves the final processed data in `processed_data`.

3. **`train_model.py`**

   - Loads the processed data.
   - Trains machine learning models for price prediction.
   - Saves the trained models and encoders in `models`.

4. **`app.py`**
   - Loads processed data and trained models.
   - Provides an interactive dashboard for:
     - Exploratory Data Analysis
     - Price Prediction
     - Recommendations

## Features

1. **Price Prediction**:
   Predicts the nightly price of an Airbnb listing given parameters like number of bedrooms, bathrooms, accommodates, etc.

2. **Recommendations**:

   - **By Description**: Uses TF-IDF similarity on listing names/descriptions.
   - **By Location**: Recommends nearest listings based on latitude/longitude.
   - **By Neighbourhood**: Filters listings within the same neighborhood.
   - **By Room Type**: Shows listings of the selected room type.

3. **Interactive EDA**:
   - Room type distribution
   - Price distribution across neighborhoods
   - Availability vs price trends

---

## Getting Started

### Prerequisites

- Python 3.7+ (recommended 3.9 or higher)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jnan-py/Air-BNB-analysis.git
   cd Air-BNB-analysis
   ```


2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

4. **Data Setup**:

   - Please Extract the CSV files from the zip folders provided `raw_data.zip` , `processed_data.zip` , `data.zip`
   - Place raw CSV files in the `raw_data` folder.
   - Run the data cleaning scripts (`clean_airbnb_data.py` and `prepare_data.py`) if you need to regenerate the processed data.

5. **Train the Model** (optional if you want to retrain):
   ```bash
   python train_model.py
   ```
   This will generate the model and encoder files in the `models` folder.

### Usage

- **Run the Streamlit App**:
  ```bash
  streamlit run app.py
  ```
  - Open the local URL that appears in the terminal (usually http://localhost:8501).

---

## File Descriptions

- **`clean_airbnb_data.py`**: Cleans the raw Airbnb CSV files (removes outliers, handles missing values, etc.).
- **`prepare_data.py`**: Further processes the cleaned data (feature engineering, transformations) and saves them as final CSVs.
- **`train_model.py`**: Trains a price prediction model and saves `.pkl` files to `models`.
- **`app.py`**: Main Streamlit application that loads data/models, visualizes EDA, performs predictions, and provides recommendations.

---

## Contributing

Contributions are welcome! If you would like to improve or add features:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

---
