import streamlit as st
import pandas as pd
import pickle
import joblib
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import uuid
import os
import base64
import datetime

st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")

# Background image
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Use forward slashes for paths or os.path.join
image_path = r"D:\Projects\Mini_Projects\Clickstream_customer_conversion\Image\black-friday-elements-assortment.jpg"

if os.path.exists(image_path):
    img_base64 = img_to_base64(image_path)
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.5)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

# Mappings for categorical variables
MAPPINGS = {
    'country': {
        1: 'Australia', 2: 'Austria', 3: 'Belgium', 4: 'British Virgin Islands', 5: 'Cayman Islands',
        6: 'Christmas Island', 7: 'Croatia', 8: 'Cyprus', 9: 'Czech Republic', 10: 'Denmark',
        11: 'Estonia', 12: 'unidentified', 13: 'Faroe Islands', 14: 'Finland', 15: 'France',
        16: 'Germany', 17: 'Greece', 18: 'Hungary', 19: 'Iceland', 20: 'India', 21: 'Ireland',
        22: 'Italy', 23: 'Latvia', 24: 'Lithuania', 25: 'Luxembourg', 26: 'Mexico', 27: 'Netherlands',
        28: 'Norway', 29: 'Poland', 30: 'Portugal', 31: 'Romania', 32: 'Russia', 33: 'San Marino',
        34: 'Slovakia', 35: 'Slovenia', 36: 'Spain', 37: 'Sweden', 38: 'Switzerland', 39: 'Ukraine',
        40: 'United Arab Emirates', 41: 'United Kingdom', 42: 'USA', 43: 'biz (.biz)', 44: 'com (.com)',
        45: 'int (.int)', 46: 'net (.net)', 47: 'org (*.org)'
    },
    'category': {1: 'Trousers', 2: 'Skirts', 3: 'Blouses', 4: 'Sales'},
    'color': {
        1: 'beige', 2: 'black', 3: 'blue', 4: 'brown', 5: 'burgundy', 6: 'gray', 7: 'green',
        8: 'navy blue', 9: 'many colors', 10: 'olive', 11: 'pink', 12: 'red', 13: 'violet', 14: 'white'
    },
    'location': {
        1: 'top left', 2: 'top in the middle', 3: 'top right',
        4: 'bottom left', 5: 'bottom in the middle', 6: 'bottom right'
    },
    'model': {1: 'Face', 2: 'Profile'},
    'page': {1: "Home", 2: "Category", 3: "Product", 4: "Cart", 5: "Checkout"}
}

# Cache model loading
@st.cache_resource
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Failed to load model from {file_path}: {str(e)}")
        return None

# Load models and preprocessor (update paths as needed)
class_model = load_model(r"D:\Projects\Mini_Projects\Clickstream_customer_conversion\Pickles\best_model_class.pkl")
reg_model = load_model(r"D:\Projects\Mini_Projects\Clickstream_customer_conversion\Pickles\best_model_reg.pkl")
clust_model = load_model(r"D:\Projects\Mini_Projects\Clickstream_customer_conversion\Pickles\best_model_clust.pkl")
preprocessor = load_model(r"D:\Projects\Mini_Projects\Clickstream_customer_conversion\Pickles\preprocessed_data.pkl")

# Cache fallback KMeans model
@st.cache_resource
def get_fallback_kmeans(n_clusters=4):
    return KMeans(n_clusters=n_clusters, random_state=42)

# Debug model features
def debug_model_features(model, df):
    with st.expander("Show Model Debugging Information"):
        if hasattr(model, 'feature_names_in_'):
            st.write(f"Model expects {len(model.feature_names_in_)} features:")
            st.write(model.feature_names_in_)
            common_features = [col for col in model.feature_names_in_ if col in df.columns]
            st.write(f"Found {len(common_features)}/{len(model.feature_names_in_)} expected features in data")
            if len(common_features) < len(model.feature_names_in_):
                missing = [f for f in model.feature_names_in_ if f not in df.columns]
                st.write("Missing features:", missing)
        elif hasattr(model, 'n_features_in_'):
            st.write(f"Model expects {model.n_features_in_} features but doesn't provide feature names")
            st.write("Available dataframe features:", df.columns.tolist())
        else:
            st.write("Model doesn't provide feature information")
            st.write("Available dataframe features:", df.columns.tolist())

# Feature engineering
def compute_features(df):
    df_features = df.copy()
    total_clicks = df_features.groupby('session_id')['order'].count()
    avg_price = df_features.groupby('session_id')['price'].mean()
    unique_products = df_features.groupby('session_id')['page2_clothing_model'].nunique()
    browsing_depth = df_features.groupby('session_id')['page'].max()
    df_features['total_clicks'] = df_features['session_id'].map(total_clicks)
    df_features['avg_price'] = df_features['session_id'].map(avg_price)
    df_features['unique_products'] = df_features['session_id'].map(unique_products)
    df_features['browsing_depth'] = df_features['session_id'].map(browsing_depth)
    df_features['date'] = pd.to_datetime(df_features[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
    df_features['weekday'] = df_features['date'].dt.dayofweek
    df_features['weekend'] = (df_features['weekday'] >= 5).astype(int)
    median_price = df_features['price'].median() if not df_features['price'].empty else 100
    df_features['high_price_preference'] = (df_features['price'] > median_price).astype(int)
    df_features.drop(columns=['date'], inplace=True)
    return df_features

# Sidebar navigation
st.sidebar.title("Pages")
page = st.sidebar.radio("Go to", ["Upload Data", "Manual Entry", "Classification", "Regression", "Clustering"])

# Initialize session state
if "df_features" not in st.session_state:
    st.session_state.df_features = None

# Reset data button
if st.sidebar.button("Reset Data"):
    st.session_state.df_features = None
    st.sidebar.success("Data reset successfully!")

# Help section
st.sidebar.markdown("### Help")
st.sidebar.write("Upload a CSV file or enter data manually, compute features, and use the Classification, Regression, or Clustering pages to analyze customer behavior.")

if page == "Upload Data":
    st.title("Upload CSV File")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())
        
        if st.button('Compute New Features'):
            st.session_state.df_features = compute_features(data)
            st.write("Feature Engineered Data")
            st.write(st.session_state.df_features.head())

        if st.button("Save Data"):
            if st.session_state.df_features is not None:
                st.success("Data saved successfully for further analysis.")
            else:
                st.warning("Please compute features before saving.")

elif page == "Manual Entry":
    st.title("Enter Data Manually for Analysis")
    current_year = datetime.datetime.now().year
    year = st.selectbox("Select Year", list(range(2000, 2026)), index=(current_year - 2000))
    # year = st.selectbox("Select Year", [2008])
    month = st.slider("Enter Month", 1, 12, 1)
    day = st.slider("Enter Day", 1, 31, 1)
    session_id = st.number_input("Select Session ID", min_value=1000, max_value=99999)
    order = st.slider("Enter Click Sequence", 1, 200, 1)
    country = st.selectbox("Select Country", list(MAPPINGS['country'].values()))
    category = st.selectbox("Select Category", list(MAPPINGS['category'].values()))
    model_number = st.text_input("Enter Model Number")
    color = st.selectbox("Select Colour", list(MAPPINGS['color'].values()))
    location = st.selectbox("Select Photo Location", list(MAPPINGS['location'].values()))
    model = st.selectbox("Select Model Photo", list(MAPPINGS['model'].values()))
    price = st.slider("Enter Price ($)", 10, 200, 10)
    page = st.selectbox("Select Page", list(MAPPINGS['page'].values()))

    # Map inputs to numeric values
    country = [k for k, v in MAPPINGS['country'].items() if v == country][0]
    category = [k for k, v in MAPPINGS['category'].items() if v == category][0]
    color = [k for k, v in MAPPINGS['color'].items() if v == color][0]
    location = [k for k, v in MAPPINGS['location'].items() if v == location][0]
    model = [k for k, v in MAPPINGS['model'].items() if v == model][0]
    page = [k for k, v in MAPPINGS['page'].items() if v == page][0]

    user_input = pd.DataFrame([{
        "year": year, "month": month, "day": day, "session_id": session_id, "order": order,
        "country": country, "page1_main_category": category, "page2_clothing_model": model_number,
        "colour": color, "location": location, "model_photography": model, "price": price, "page": page
    }])

    if st.button("View Entered Data"):
        st.write("Entered Data")
        st.write(user_input)

    if st.button("Compute New Features"):
        st.session_state.df_features = compute_features(user_input)
        st.write("Feature Engineered Data")
        st.write(st.session_state.df_features)
    
    if st.button("Save Data"):
        if st.session_state.df_features is not None:
            st.success("Data saved successfully for further analysis.")
        else:
            st.warning("Please compute features before saving.")

elif page == "Classification":
    st.title("Customer Purchase Prediction")
    st.write("Predict if a customer will complete a purchase.")
    
    if st.session_state.df_features is not None:
        st.write("### Feature Engineered Data")
        st.write(st.session_state.df_features.head())
        
        if class_model is None:
            st.error("Classification model not loaded. Please check the model file.")
        else:
            debug_model_features(class_model, st.session_state.df_features)
            try:
                required_features = getattr(class_model, 'feature_names_in_', None)
                if required_features is not None:
                    common_features = [col for col in required_features if col in st.session_state.df_features.columns]
                    if len(common_features) == len(required_features):
                        prediction_data = st.session_state.df_features[common_features]
                        if preprocessor is not None:
                            prediction_data = preprocessor.transform(prediction_data)
                        predictions = class_model.predict(prediction_data)
                        st.session_state.df_features["Prediction"] = predictions

                        st.write("### Classification Results")
                        display_cols = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'high_price_preference', 'Prediction']
                        st.write(st.session_state.df_features[display_cols])

                        fig = px.histogram(st.session_state.df_features, x='Prediction', title='Distribution of Purchase Predictions')
                        st.plotly_chart(fig)
                    else:
                        st.error(f"Missing features: {[f for f in required_features if f not in common_features]}")
                        st.write("Please ensure all required features are included in the dataset.")
                else:
                    st.warning("Model does not provide feature information. Using basic features.")
                    basic_features = ['total_clicks', 'avg_price', 'unique_products', 'browsing_depth', 'page', 'price', 'high_price_preference', 'weekday', 'weekend']
                    available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
                    if available_features:
                        df_for_prediction = st.session_state.df_features[available_features]
                        scaler = StandardScaler()
                        numeric_features = df_for_prediction.select_dtypes(include=['float64', 'int64']).columns
                        df_for_prediction[numeric_features] = scaler.fit_transform(df_for_prediction[numeric_features])
                        predictions = class_model.predict(df_for_prediction)
                        st.session_state.df_features["Prediction"] = predictions

                        st.write("### Classification Results (Basic Features)")
                        st.write(st.session_state.df_features[display_cols])
                        fig = px.histogram(st.session_state.df_features, x='Prediction', title='Distribution of Purchase Predictions (Basic Features)')
                        st.plotly_chart(fig)
                    else:
                        st.error("No compatible features found for prediction.")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Please check the data and model compatibility.")
    else:
        st.warning("Please upload or manually enter data before running classification.")

elif page == "Regression":
    st.title("Customer Spending Prediction")
    st.write("Predict how much a customer will spend.")
    
    if st.session_state.df_features is not None:
        st.write("### Feature Engineered Data")
        st.write(st.session_state.df_features.head())
        
        if reg_model is None:
            st.error("Regression model not loaded. Please check the model file.")
        else:
            debug_model_features(reg_model, st.session_state.df_features)
            try:
                required_features = getattr(reg_model, 'feature_names_in_', None)
                if required_features is not None:
                    common_features = [col for col in required_features if col in st.session_state.df_features.columns]
                    if len(common_features) == len(required_features):
                        prediction_data = st.session_state.df_features[common_features]
                        if preprocessor is not None:
                            prediction_data = preprocessor.transform(prediction_data)
                        predictions = reg_model.predict(prediction_data)
                        st.session_state.df_features["Predicted_Spending"] = predictions

                        st.write("### Regression Results")
                        display_cols = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'high_price_preference', 'Predicted_Spending']
                        st.write(st.session_state.df_features[display_cols])

                        if 'total_clicks' in st.session_state.df_features.columns:
                            fig = px.scatter(st.session_state.df_features, x='total_clicks', y='Predicted_Spending',
                                            color='high_price_preference', hover_data=['page1_main_category'],
                                            title='Predicted Spending vs Total Clicks')
                            st.plotly_chart(fig)
                    else:
                        st.error(f"Missing features: {[f for f in required_features if f not in common_features]}")
                        st.write("Please ensure all required features are included in the dataset.")
                else:
                    st.warning("Model does not provide feature information. Using basic features.")
                    basic_features = ['total_clicks', 'avg_price', 'unique_products', 'browsing_depth', 'page', 'price', 'high_price_preference', 'weekday', 'weekend']
                    available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
                    if available_features:
                        df_for_prediction = st.session_state.df_features[available_features]
                        scaler = StandardScaler()
                        numeric_features = df_for_prediction.select_dtypes(include=['float64', 'int64']).columns
                        df_for_prediction[numeric_features] = scaler.fit_transform(df_for_prediction[numeric_features])
                        predictions = reg_model.predict(df_for_prediction)
                        st.session_state.df_features["Predicted_Spending"] = predictions

                        st.write("### Regression Results (Basic Features)")
                        st.write(st.session_state.df_features[display_cols])
                        fig = px.histogram(st.session_state.df_features, x='Predicted_Spending', nbins=20,
                                         title='Distribution of Predicted Spending')
                        st.plotly_chart(fig)
                    else:
                        st.error("No compatible features found for prediction.")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Please check the data and model compatibility.")
    else:
        st.warning("Please upload or manually enter data before running regression.")

elif page == "Clustering":
    st.title("Customer Segmentation")
    st.write("Group customers into segments based on their behavior.")
    
    if st.session_state.df_features is not None:
        st.write("### Feature Engineered Data")
        st.write(st.session_state.df_features.head())
        
        if clust_model is None:
            st.error("Clustering model not loaded. Using fallback KMeans model.")
            clust_model = get_fallback_kmeans()
        
        debug_model_features(clust_model, st.session_state.df_features)
        try:
            required_features = getattr(clust_model, 'feature_names_in_', None)
            if required_features is not None:
                common_features = [col for col in required_features if col in st.session_state.df_features.columns]
                if len(common_features) == len(required_features):
                    clustering_data = st.session_state.df_features[common_features]
                    if preprocessor is not None:
                        clustering_data = preprocessor.transform(clustering_data)
                    clusters = clust_model.predict(clustering_data)
                    st.session_state.df_features["Cluster"] = clusters

                    st.write("### Clustering Results")
                    display_cols = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'high_price_preference', 'Cluster']
                    st.write(st.session_state.df_features[display_cols])

                    cluster_counts = st.session_state.df_features["Cluster"].value_counts().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    fig1 = px.pie(cluster_counts, names="Cluster", values="Count", title="Customer Segment Distribution")
                    st.plotly_chart(fig1)

                    if 'total_clicks' in st.session_state.df_features.columns and 'avg_price' in st.session_state.df_features.columns:
                        selected_clusters = st.multiselect("Select Clusters to Display", st.session_state.df_features["Cluster"].unique(), default=st.session_state.df_features["Cluster"].unique())
                        fig2 = px.scatter(st.session_state.df_features[st.session_state.df_features["Cluster"].isin(selected_clusters)],
                                        x='total_clicks', y='avg_price', color='Cluster', hover_data=['page1_main_category'],
                                        title="Cluster Distribution by Total Clicks and Average Price")
                        st.plotly_chart(fig2)

                    st.write("### Cluster Characteristics")
                    numeric_cols = st.session_state.df_features.select_dtypes(include=['float64', 'int64']).columns
                    cluster_profiles = st.session_state.df_features.groupby('Cluster')[numeric_cols].mean()
                    st.write(cluster_profiles)
                else:
                    st.error(f"Missing features: {[f for f in required_features if f not in common_features]}")
                    st.write("Please ensure all required features are included in the dataset.")
            else:
                st.warning("Model does not provide feature information. Using basic features.")
                basic_features = ['total_clicks', 'avg_price', 'unique_products', 'browsing_depth', 'price', 'high_price_preference']
                available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
                if available_features:
                    df_for_clustering = st.session_state.df_features[available_features]
                    scaler = StandardScaler()
                    numeric_features = df_for_clustering.select_dtypes(include=['float64', 'int64']).columns
                    df_for_clustering[numeric_features] = scaler.fit_transform(df_for_clustering[numeric_features])
                    
                    temp_model = get_fallback_kmeans()
                    temp_model.fit(df_for_clustering)
                    clusters = temp_model.labels_
                    st.session_state.df_features["Cluster"] = clusters

                    st.write("### Clustering Results (Basic Features)")
                    st.write(st.session_state.df_features[display_cols])

                    st.write("### Cluster Centers (Standardized Features)")
                    cluster_centers_df = pd.DataFrame(temp_model.cluster_centers_, columns=available_features)
                    st.write(cluster_centers_df)

                    cluster_counts = st.session_state.df_features["Cluster"].value_counts().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    fig = px.pie(cluster_counts, names="Cluster", values="Count", title="Customer Segment Distribution")
                    st.plotly_chart(fig)

                    st.write("### Cluster Characteristics")
                    cluster_means = st.session_state.df_features.groupby('Cluster')[available_features].mean()
                    st.write(cluster_means)
                else:
                    st.error("No compatible features found for clustering.")
        except Exception as e:
            st.error(f"Clustering failed: {str(e)}")
            st.write("Please check the data and model compatibility.")
    else:
        st.warning("Please upload or manually enter data before running clustering.")

# Save predictions for Classification, Regression, Clustering
if page in ["Classification", "Regression", "Clustering"]:
    if st.button("Save Predictions"):
        if st.session_state.df_features is not None:
            file_name = f"predictions_{uuid.uuid4()}.csv"
            st.session_state.df_features.to_csv(file_name, index=False)
            st.download_button(
                label="Download Predictions",
                data=st.session_state.df_features.to_csv(index=False),
                file_name=file_name,
                mime="text/csv"
            )
            st.success("Predictions saved successfully!")
        else:
            st.warning("No predictions available to save.")
