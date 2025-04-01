import os
import pandas as pd
import pickle
import numpy as np
import streamlit as st
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def compute_tfidf_matrix(descriptions):    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(descriptions)
    return tfidf, tfidf_matrix

def get_recommendations_by_description(user_input, tfidf, tfidf_matrix, df, top_n=5):    
    input_vec = tfidf.transform([user_input])    
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()    
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

def haversine(lon1, lat1, lon2, lat2): 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def get_recommendations_by_location(listing_id, df, top_n=5):
    target = df[df['id'] == listing_id]
    if target.empty:
        st.error("Listing ID not found.")
        return pd.DataFrame()
    target_lat = target.iloc[0]['latitude']
    target_lon = target.iloc[0]['longitude']    
    df['distance_km'] = df.apply(lambda row: haversine(target_lon, target_lat, row['longitude'], row['latitude']), axis=1)
    recommendations = df[df['id'] != listing_id].sort_values('distance_km').head(top_n)
    return recommendations

DATA_FOLDER = "processed_data"
MODEL_FOLDER = "models"

st.set_page_config(page_icon=":house:", page_title="Air - BNB Data Analysis", layout="wide")

if os.path.exists(DATA_FOLDER):
    city_files = [f.replace("_processed.csv", "") for f in os.listdir(DATA_FOLDER) if f.endswith("_processed.csv")]
else:
    st.error(f"❌ Data folder not found: {DATA_FOLDER}")
    city_files = []

st.sidebar.title("🏙 Select a City")
selected_city = st.sidebar.selectbox("Choose a city:", ["Select a city"] + city_files)

if selected_city != "Select a city":
    data_path = os.path.join(DATA_FOLDER, f"{selected_city}_processed.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        with st.spinner("Getting all the necessary details..."):
            model_path = os.path.join(MODEL_FOLDER, f"{selected_city}_model.pkl")
            encoder_path = os.path.join(MODEL_FOLDER, f"{selected_city}_encoder.pkl")
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                    model = model_data["model"]
                    feature_names = model_data["features"]

                with open(encoder_path, "rb") as f:
                    encoder = pickle.load(f)

                st.sidebar.subheader("💸 Predict Property Price")
                accommodates = st.sidebar.slider("Accommodates", 1, 10, 2)
                bedrooms = st.sidebar.slider("Bedrooms", 0, 5, 1)
                bathrooms = st.sidebar.slider("Bathrooms", 0, 5, 1)
                minimum_nights = st.sidebar.slider("Minimum Nights", 1, 30, 2)
                
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                categorical_inputs = {}
                if categorical_cols:
                    for col in categorical_cols:
                        options = df[col].unique().tolist()
                        selected_value = st.sidebar.selectbox(f"{col.replace('_', ' ').title()}", options)
                        categorical_inputs[col] = selected_value

                    cat_values = pd.DataFrame([categorical_inputs])
                    encoded_cat = encoder.transform(cat_values)
                    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))
                else:
                    encoded_df = pd.DataFrame()
            
                if categorical_cols:
                    st.write("Categorical Inputs DataFrame:", cat_values)
                    st.write("Encoder Feature Names:", encoder.get_feature_names_out(categorical_cols))

                input_data = pd.DataFrame(
                    [[accommodates, bedrooms, bathrooms, minimum_nights]],
                    columns=["accommodates", "bedrooms", "bathrooms", "minimum_nights"],
                )

                full_input = pd.DataFrame(columns=feature_names)
                for col in input_data.columns:
                    full_input[col] = input_data[col].values
                if not encoded_df.empty:
                    for col in encoded_df.columns:
                        full_input[col] = encoded_df[col].values
                full_input = full_input.reindex(columns=feature_names, fill_value=0)
                full_input = full_input.fillna(0).infer_objects(copy=False)

                if full_input.shape[1] != len(feature_names):
                    st.sidebar.error(f"❌ Feature Mismatch! Expected {len(feature_names)} features, got {full_input.shape[1]}.")
                else:
                    predicted_price = max(0, model.predict(full_input)[0])
                
                    st.sidebar.success(f"🏠 Predicted Price: *${predicted_price[0]*(minimum_nights*2.4 + bedrooms*1.2 + (accommodates+bathrooms)*1):.2f}*")
            else:
                st.sidebar.error("❌ Model or encoder not found for this city.")

            st.title(f"📊 {selected_city} Airbnb Analysis")
            tab_dashboard, tab_recommend = st.tabs(["Dashboard", "Recommendations"])
        
            with tab_dashboard:                
                st.write("### Dataset Overview:")
                st.write(df.describe())

                st.write("## 📊 Insights Dashboard")
                
                room_type_df = df.filter(regex="^room_type_")
                if not room_type_df.empty:
                    room_types = [room_type_df[i].sum() for i in room_type_df.columns]
                    common_room_type = room_type_df.columns[np.argmax(room_types)][10:]
                else:
                    common_room_type = "N/A"
                                
                avg_price = df["price"].mean()
                col1, col2 = st.columns(2)
                col1.metric("Average Price", f"${avg_price:.2f}")
                col2.metric("Most Common Room Type", common_room_type)
                
                if "room_type" in df.columns:
                    st.write("### Room Type Distribution")
                    room_type_counts = None
                    room_type_cols = [col for col in df.columns if col.startswith("room_type_")]
                    if room_type_cols:
                        df["room_type"] = df[room_type_cols].idxmax(axis=1).str.replace("room_type_", "")
                        room_type_counts = df["room_type"].value_counts()
                    if room_type_counts is not None and len(room_type_counts) > 0:
                        st.bar_chart(room_type_counts)
                    else:
                        st.warning("No room type data available.")
                
                neighborhood_cols = [col for col in df.columns if col.startswith("neighbourhood_")]
                if neighborhood_cols:
                    df["neighbourhood"] = df[neighborhood_cols].idxmax(axis=1).str.replace("neighbourhood_", "")
                    fig_price_dist = px.box(
                        df,
                        x="neighbourhood",
                        y="price",
                        title="Price Distribution Across Neighborhoods",
                        labels={"neighbourhood": "Neighborhood", "price": "Price"},
                        color="neighbourhood",
                    )
                    st.plotly_chart(fig_price_dist)
                
                room_type_cols = [col for col in df.columns if col.startswith("room_type_")]
                if room_type_cols:
                    df["room_type"] = df[room_type_cols].idxmax(axis=1).str.replace("room_type_", "")
                    fig_room_price = px.box(
                        df,
                        x="room_type",
                        y="price",
                        title="Room Type vs Price Comparison",
                        labels={"room_type": "Room Type", "price": "Price"},
                        color="room_type",
                    )
                    st.plotly_chart(fig_room_price)
                
                if "availability_365" in df.columns:
                    fig_availability = px.scatter(
                        df,
                        x="availability_365",
                        y="price",
                        title="Availability Trends Over the Year",
                        labels={"availability_365": "Availability (Days)", "price": "Price"},
                        color="neighbourhood" if "neighbourhood" in df.columns else None,
                    )
                    st.plotly_chart(fig_availability)

            with tab_recommend:
                st.header("🏠 Property Recommendations")            
                        
                data_path_ = os.path.join('raw_data', f"{selected_city}_listings.csv")
                if os.path.exists(data_path_):
                    sdf = pd.read_csv(data_path_)

                rec_method = st.radio("Select Recommendation Type", ("By Description", "By Location using Listing ID", "By Neighbourhood", "By Room Type"))
                
                if rec_method == "By Description":
                    user_input_desc = st.text_input("Enter a description snippet or listing description:", value="Long Windos, office view")
                    n = st.number_input("Enter the Recommendations to Display : ", min_value=1, max_value=10, value=1)

                    if st.button("Get Recommendations (Description)", key="desc_rec"):
                        with st.spinner("Getting Recommendations.."):
                            if user_input_desc:                            
                                descriptions = sdf["name"].fillna("")
                                tfidf, tfidf_matrix = compute_tfidf_matrix(descriptions)
                                recs_desc = get_recommendations_by_description(user_input_desc, tfidf, tfidf_matrix, sdf, top_n=n)
                                st.write("### Recommendations based on Description:")
                                st.dataframe(recs_desc.reset_index(drop=True))
                            else:
                                st.warning("Please enter a description snippet.")

                elif rec_method == "By Location using Listing ID":
                    user_input_id = st.selectbox("Choose id for Location", options = sdf['id'].unique())
                    n = st.number_input("Enter the Recommendations to Display : ", min_value=1, max_value=10, value=1)
                    if st.button("Get Recommendations (Location)", key="loc_rec"):
                        with st.spinner("Getting Recommendations.."):
                            try:
                                listing_id = int(user_input_id)
                                recs_loc = get_recommendations_by_location(listing_id, sdf, top_n=n)
                                if not recs_loc.empty:
                                    st.write("### Recommendations based on Geographic Proximity (lon, lat):")
                                    st.dataframe(recs_loc.reset_index(drop=True))
                            except ValueError:
                                st.error("Please enter a valid numeric Listing ID.")

                elif rec_method == "By Neighbourhood":
                    user_input_neighbourhood = st.selectbox("Choose Neighbourhood", options = sdf['neighbourhood'].unique())
                    n = st.number_input("Enter the Recommendations to Display : ", min_value=1, max_value=10, value=1)
                    if st.button("Get Recommendations (Neighbourhood)", key="neigh_rec"):
                        with st.spinner("Getting Recommendations.."):
                            try:
                                st.write("### Recommendations based on Neighbourhood:")
                                n_df = sdf[sdf['neighbourhood']==user_input_neighbourhood].head(n)
                                st.write(n_df)
                            except Exception as e:
                                st.warning(f"Cannot fetch data at this point of time")

                else:
                    user_input_rt = st.selectbox("Choose Room Type", options = sdf['room_type'].unique())
                    n = st.number_input("Enter the Recommendations to Display : ", min_value=1, max_value=10, value=1)
                    if st.button("Get Recommendations (Room Type)", key="rt_rec"):
                        with st.spinner("Getting Recommendations.."):
                            try:
                                st.write("### Recommendations based on Room Type:")
                                rt_df = sdf[sdf['room_type']==user_input_rt].head(n)
                                st.write(rt_df)
                            except Exception as e:
                                st.warning(f"Cannot fetch data at this point of time")
    else:
        st.error(f"❌ Dataset not found for {selected_city}.")
else:
    st.title(f"📊 Airbnb Analysis")
    st.markdown("""
    Welcome to the Airbnb Data Analysis & Recommendation System! This application allows you to:
    - 🔍 **Analyze Airbnb listings** across various cities.
    - 📊 **Visualize trends** in price, room type distribution, and availability.
    - 💰 **Predict rental prices** based on listing features.
    - 🏠 **Get personalized recommendations** based on location, description, and room type.
    """)
    
    st.markdown("#### 🔗 Explore the Features")

    st.markdown("""
    - **📊 Data Dashboard:** Visualize insights from Airbnb listings.
    - **💰 Price Prediction:** Estimate rental prices based on listing details.
    - **🏠 Property Recommendations:** Find similar listings based on various factors.
    """)
    
    st.markdown("#### 🚀 Get Started!")
    st.write("Select a city from Sidebar, to get started")
    
    
