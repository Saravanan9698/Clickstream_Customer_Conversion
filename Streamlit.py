import streamlit as st
import pandas as pd
import pickle
import os
import base64
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Background image
def img_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Background image not found at {image_path}. Proceeding without background.")
        return None

image_path = os.path.join(BASE_DIR, "Image", "black-friday-elements-assortment.jpg")
if os.path.exists(image_path):
    img_base64 = img_to_base64(image_path)
    if img_base64:
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.5)), url('data:image/jpeg;base64,{img_base64}');
                background-size: cover;
                background-position: center;
                color: white;
            }}
            h1, h2, h3 {{
                color: #FFD700;
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }}
            </style>
        """, unsafe_allow_html=True)

# Cache pickle loading
@st.cache_resource
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load {file_path}: {e}. Ensure the file exists and is not corrupted.")
        return None

# Load models
class_model = load_pickle(os.path.join(BASE_DIR, "Pickles", "best_model_class.pkl"))
reg_model = load_pickle(os.path.join(BASE_DIR, "Pickles", "best_model_reg.pkl"))
clust_model = load_pickle(os.path.join(BASE_DIR, "Pickles", "best_model_clust.pkl"))
preprocessor = load_pickle(os.path.join(BASE_DIR, "Pickles", "preprocessed_data.pkl"))

# Debug model features
def debug_model_features(model, df, show_debug=False):
    if not show_debug:
        return
    st.write("### Model Debugging Information")
    if hasattr(model, 'feature_names_in_'):
        expected = model.feature_names_in_
        st.write(f"Model expects {len(expected)} features: {expected}")
        common = [col for col in expected if col in df.columns]
        st.write(f"Found {len(common)}/{len(expected)} features in data")
        if common != expected:
            st.write("Missing features:", [f for f in expected if f not in df.columns])
    else:
        st.write("Model doesn't provide feature names")
    st.write("Available dataframe features:", df.columns.tolist())

# Feature engineering
@st.cache_data
def feature_engineering(df):
    df = df.copy()
    if 'price_2' in df.columns:
        df['price_ratio'] = df['price'] / df['price_2']
    df['clicks_per_product'] = df['total_clicks'] / df['unique_products'].replace(0, 1)
    df['price_per_click'] = df['avg_price'] / df['total_clicks'].replace(0, 1)
    df['price_browsing_interaction'] = df['avg_price'] * df['browsing_depth']
    df['weekend_clicks'] = df['weekend'] * df['total_clicks']
    skewed_cols = ['total_clicks', 'unique_products', 'avg_price']
    for col in skewed_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    return df

# Align features with model's expected features
def align_features(X_new, expected_features, numeric_cols=None):
    X_new = X_new.copy()
    missing_features = [f for f in expected_features if f not in X_new.columns]
    for feature in missing_features:
        if numeric_cols and feature in numeric_cols:
            X_new[feature] = X_new[numeric_cols].mean().get(feature, 0)
        else:
            X_new[feature] = 0
    X_new = X_new.drop(columns=[f for f in X_new.columns if f not in expected_features], errors='ignore')
    X_new = X_new[expected_features]
    return X_new, missing_features

# Validate data
def validate_data(df, required_cols=None):
    if df.empty:
        return False, "Dataset is empty."
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    return True, ""

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Manual Entry", "Classification", "Regression", "Clustering", "Help"])

# Initialize session state
if "df_features" not in st.session_state:
    st.session_state.df_features = None
if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = []

# Reset session state
if st.sidebar.button("Reset Data"):
    st.session_state.df_features = None
    st.session_state.manual_entries = []
    st.success("Session data reset successfully!")

# Page: Home
if page == "Home":
    st.title("Welcome to Clickstream Customer Conversion Analysis")
    st.markdown("""
    ### Unlock Insights from Customer Behavior
    This application analyzes **clickstream data** to help e-commerce businesses understand customer behavior, predict conversions, and optimize strategies. Powered by machine learning, it provides actionable insights through classification, regression, and clustering.

    #### What You Can Do:
    - **Predict Purchases**: Determine the likelihood of a customer completing a purchase.
    - **Estimate Spending**: Forecast how much customers are likely to spend.
    - **Segment Customers**: Group customers based on browsing and purchasing patterns.
    - **Visualize Insights**: Explore interactive charts and graphs to understand trends.

    #### Get Started:
    1. **Upload Data**: Use a CSV file with clickstream data (e.g., session_id, price, page).
    2. **Enter Manually**: Input data point-by-point for quick analysis.
    3. **Analyze**: Choose Classification, Regression, or Clustering to generate insights.
    4. **Explore**: View visualizations and download predictions.

    Ready to dive in? Navigate to the **Upload Data** or **Manual Entry** page to begin!
    """)

    st.image(image_path if os.path.exists(image_path) else "https://via.placeholder.com/800x400", caption="Analyze Customer Behavior")
    st.markdown("""
    **Need Help?** Check the [Help](#help) page for detailed instructions and troubleshooting.
    """)

# Page: Upload Data
elif page == "Upload Data":
    st.title("Upload CSV File")
    st.markdown("Upload a CSV file containing clickstream data to compute features and perform analysis.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            data = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(data.head())

        required_cols = ['session_id', 'price', 'page2_clothing_model', 'year', 'month', 'day']
        valid, error_msg = validate_data(data, required_cols)
        if not valid:
            st.error(error_msg)
        elif st.button('Compute New Features'):
            with st.spinner("Computing features..."):
                df_features = data.copy()
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
                df_features = feature_engineering(df_features)
                st.session_state.df_features = df_features
                st.write("### Feature Engineered Data")
                st.write(df_features.head())

        if st.button("Save Data"):
            if st.session_state.df_features is not None:
                st.success("Data saved successfully for further analysis.")
            else:
                st.warning("Please compute features before saving.")

# Page: Manual Entry
elif page == "Manual Entry":
    st.title("Manual Data Entry")
    st.markdown("Enter clickstream data manually for analysis.")

    country_mappings = {
        1: 'Australia', 2: 'Austria', 3: 'Belgium', 4: 'British Virgin Islands', 5: 'Cayman Islands',
        6: 'Christmas Island', 7: 'Croatia', 8: 'Cyprus', 9: 'Czech Republic', 10: 'Denmark',
        11: 'Estonia', 12: 'unidentified', 13: 'Faroe Islands', 14: 'Finland', 15: 'France',
        16: 'Germany', 17: 'Greece', 18: 'Hungary', 19: 'Iceland', 20: 'India', 21: 'Ireland',
        22: 'Italy', 23: 'Latvia', 24: 'Lithuania', 25: 'Luxembourg', 26: 'Mexico', 27: 'Netherlands',
        28: 'Norway', 29: 'Poland', 30: 'Portugal', 31: 'Romania', 32: 'Russia', 33: 'San Marino',
        34: 'Slovakia', 35: 'Slovenia', 36: 'Spain', 37: 'Sweden', 38: 'Switzerland', 39: 'Ukraine',
        40: 'United Arab Emirates', 41: 'United Kingdom', 42: 'USA', 43: 'biz (.biz)', 44: 'com (.com)',
        45: 'int (.int)', 46: 'net (.net)', 47: 'org (*.org)'
    }
    category_mappings = {1: 'Trousers', 2: 'Skirts', 3: 'Blouses', 4: 'Sales'}
    color_mappings = {
        1: 'beige', 2: 'black', 3: 'blue', 4: 'brown', 5: 'burgundy', 6: 'gray', 7: 'green', 8: 'navy blue',
        9: 'many colors', 10: 'olive', 11: 'pink', 12: 'red', 13: 'violet', 14: 'white'
    }
    location_mappings = {
        1: 'top left', 2: 'top in the middle', 3: 'top right', 4: 'bottom left', 5: 'bottom in the middle', 6: 'bottom right'
    }
    model_mappings = {1: 'Face', 2: 'Profile'}
    page_mappings = {1: "Home", 2: "Category", 3: "Product", 4: "Cart", 5: "Checkout"}

    with st.form("manual_entry_form"):
        year = st.selectbox("Select Year", [2008])
        month = st.slider("Enter Month", 1, 12, 1)
        day = st.slider("Enter Day", 1, 31, 1)
        session_id = st.number_input("Select Session ID", min_value=1000, max_value=99999, value=1000)
        order = st.slider("Enter Click Sequence", 1, 200, 1)
        country = st.selectbox("Select Country", list(country_mappings.values()))
        category = st.selectbox("Select Category", list(category_mappings.values()))
        model_number = st.text_input("Enter Model Number")
        color = st.selectbox("Select Colour", list(color_mappings.values()))
        location = st.selectbox("Select Photo Location", list(location_mappings.values()))
        model = st.selectbox("Select Model Photo", list(model_mappings.values()))
        price = st.slider("Enter Price ($)", 10, 200, 10)
        page = st.selectbox("Select Page", list(page_mappings.values()))
        submit = st.form_submit_button("Add Entry")

        if submit:
            country_key = list(country_mappings.keys())[list(country_mappings.values()).index(country)]
            category_key = list(category_mappings.keys())[list(category_mappings.values()).index(category)]
            color_key = list(color_mappings.keys())[list(color_mappings.values()).index(color)]
            location_key = list(location_mappings.keys())[list(location_mappings.values()).index(location)]
            model_key = list(model_mappings.keys())[list(model_mappings.values()).index(model)]
            page_key = list(page_mappings.keys())[list(page_mappings.values()).index(page)]
            entry = {
                "year": year, "month": month, "day": day, "session_id": session_id, "order": order,
                "country": country_key, "page1_main_category": category_key, "page2_clothing_model": model_number,
                "colour": color_key, "location": location_key, "model_photography": model_key, "price": price, "page": page_key
            }
            st.session_state.manual_entries.append(entry)
            st.success("Entry added! Add more or proceed to compute features.")

    if st.session_state.manual_entries:
        user_input = pd.DataFrame(st.session_state.manual_entries)
        st.write("### Entered Data")
        st.write(user_input)

        if st.button("Compute New Features"):
            with st.spinner("Computing features..."):
                df_features = user_input.copy()
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
                df_features = feature_engineering(df_features)
                st.session_state.df_features = df_features
                st.write("### Feature Engineered Data")
                st.write(df_features)

        if st.button("Save Data"):
            if st.session_state.df_features is not None:
                st.success("Data saved successfully for further analysis.")
            else:
                st.warning("Please compute features before saving.")

# Page: Classification
elif page == "Classification":
    st.title("Customer Purchase Prediction")
    st.markdown("Predict whether a customer will complete a purchase based on their clickstream behavior.")

    if st.session_state.df_features is not None and class_model:
        st.write("### Feature Engineered Data")
        st.write(st.session_state.df_features.head())

        show_debug = st.checkbox("Show debug information")
        debug_model_features(class_model, st.session_state.df_features, show_debug)

        try:
            if hasattr(class_model, 'feature_names_in_'):
                common_features = [col for col in class_model.feature_names_in_ if col in st.session_state.df_features.columns]
                if len(common_features) == len(class_model.feature_names_in_):
                    prediction_data = st.session_state.df_features[common_features]
                    if preprocessor and isinstance(preprocessor, StandardScaler):
                        numeric_cols = prediction_data.select_dtypes(include=['float64', 'int64']).columns
                        prediction_data[numeric_cols] = preprocessor.transform(prediction_data[numeric_cols])
                    with st.spinner("Making predictions..."):
                        predictions = class_model.predict(prediction_data)
                    st.session_state.df_features["Prediction"] = predictions
                    st.write("### Classification Results")
                    display_cols = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'high_price_preference', 'Prediction']
                    display_cols = [col for col in display_cols if col in st.session_state.df_features.columns]
                    st.write(st.session_state.df_features[display_cols])
                else:
                    st.warning(f"Only found {len(common_features)} of {len(class_model.feature_names_in_)} required features.")
                    raise ValueError("Insufficient features")
            else:
                raise ValueError("Model doesn't provide feature information")
        except Exception as e:
            st.error(f"Direct prediction failed: {e}")
            st.write("### Trying alternative approach with basic features...")
            basic_features = ['total_clicks', 'avg_price', 'unique_products', 'browsing_depth', 'page', 'price', 'high_price_preference', 'weekday', 'weekend']
            available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
            if len(available_features) > 0:
                df_for_prediction = st.session_state.df_features[available_features]
                scaler = StandardScaler()
                numeric_cols = df_for_prediction.select_dtypes(include=['float64', 'int64']).columns
                df_for_prediction[numeric_cols] = scaler.fit_transform(df_for_prediction[numeric_cols])
                try:
                    with st.spinner("Making alternative predictions..."):
                        predictions = class_model.predict(df_for_prediction)
                    st.session_state.df_features["Prediction"] = predictions
                    st.write("### Classification Results (using basic features)")
                    display_cols = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'high_price_preference', 'Prediction']
                    display_cols = [col for col in display_cols if col in st.session_state.df_features.columns]
                    st.write(st.session_state.df_features[display_cols])
                except Exception as e:
                    st.error(f"Alternative prediction failed: {e}")
                    st.write("Ensure the dataset includes features like total_clicks and avg_price, or retrain the model.")
            else:
                st.error("No compatible features found for this model. Please check your data or model.")
    else:
        st.warning("Please upload or manually enter data and ensure the classification model is loaded.")

# Page: Regression
elif page == "Regression":
    st.title("Customer Spending Prediction")
    st.markdown("Predict how much a customer will spend based on their clickstream behavior.")

    if st.session_state.df_features is not None:
        st.subheader("Feature Engineered Data Preview")
        st.write(st.session_state.df_features.head())

        basic_features = [
            'total_clicks', 'avg_price', 'unique_products', 'browsing_depth',
            'page', 'price', 'high_price_preference', 'weekday', 'weekend',
            'clicks_per_product', 'price_per_click', 'price_browsing_interaction',
            'log_total_clicks', 'log_unique_products', 'log_avg_price'
        ]
        available_features = [col for col in basic_features if col in st.session_state.df_features.columns]

        if len(available_features) < 3:
            st.error(f"Insufficient features for retraining. Found only {len(available_features)} features: {available_features}")
            st.write("Ensure the dataset contains at least three features like total_clicks, avg_price, or unique_products.")
        elif 'avg_price' not in st.session_state.df_features.columns:
            st.error("Target variable 'avg_price' not found in dataset. Please ensure features are computed correctly.")
        else:
            st.write(f"Retraining model with {len(available_features)} features: {available_features}")

            df = st.session_state.df_features.copy()
            df = df.fillna(df.mean(numeric_only=True))
            
            categorical_cols = ['country', 'page1_main_category', 'colour', 'location', 'model_photography']
            existing_cats = [col for col in categorical_cols if col in df.columns]
            if existing_cats:
                df = pd.get_dummies(df, columns=existing_cats)
            
            X = df[available_features]
            y = df['avg_price']

            scaler = preprocessor if preprocessor and isinstance(preprocessor, StandardScaler) else StandardScaler()
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                if not isinstance(preprocessor, StandardScaler) or not hasattr(scaler, 'mean_'):
                    scaler.fit(X[numeric_cols])
                X[numeric_cols] = scaler.transform(X[numeric_cols])

            with st.spinner("Retraining model and making predictions..."):
                try:
                    new_model = GradientBoostingRegressor(random_state=42)
                    new_model.fit(X, y)
                    predictions = new_model.predict(X)
                    df["Predicted_Spending"] = predictions
                    st.session_state.df_features["Predicted_Spending"] = predictions

                    total_prediction_amount = np.sum(predictions)

                    st.subheader("Prediction Results")
                    display_cols = ['page1_main_category', 'colour', 'location', 
                                    'model_photography', 'page', 'high_price_preference', 'Predicted_Spending']
                    display_cols = [col for col in display_cols if col in df.columns]
                    st.write(df[display_cols])
                    st.write(f"**Total Predicted Spending**: ${total_prediction_amount:.2f}")

                    st.subheader("Visualizations")
                    if 'total_clicks' in df.columns:
                        fig = px.scatter(df, 
                                        x='total_clicks', 
                                        y='Predicted_Spending',
                                        color='high_price_preference' if 'high_price_preference' in df.columns else None,
                                        hover_data=['page1_main_category'] if 'page1_main_category' in df.columns else None,
                                        title='Predicted Spending vs Total Clicks',
                                        labels={'total_clicks': 'Total Clicks', 'Predicted_Spending': 'Predicted Spending ($)'})
                        st.plotly_chart(fig)
                    
                    fig = px.histogram(df, x='Predicted_Spending', nbins=20,
                                    title='Distribution of Predicted Spending',
                                    labels={'Predicted_Spending': 'Predicted Spending ($)'})
                    st.plotly_chart(fig)

                    df[['Predicted_Spending']].to_csv('predictions_retrained.csv', index=False)
                    with open('total_prediction_retrained.txt', 'w') as f:
                        f.write(str(total_prediction_amount))
                    st.success("Predictions completed and saved successfully!")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.write("Please check your data for inconsistencies or try recomputing features.")
    else:
        st.warning("Please upload a CSV file or manually enter data to proceed with predictions.")

# Page: Clustering
elif page == "Clustering":
    st.title("Customer Segmentation")
    st.markdown("Group customers into segments based on their clickstream behavior.")

    if st.session_state.df_features is not None and clust_model:
        st.write("### Feature Engineered Data")
        st.write(st.session_state.df_features.head())

        n_samples = st.session_state.df_features.shape[0]
        if n_samples == 0:
            st.error("Dataset is empty. Please provide data for clustering.")
        elif n_samples == 1:
            st.warning("Only one sample available. Assigning to cluster 0.")
            st.session_state.df_features["Cluster"] = [0]
            st.write("### Clustering Results")
            display_cols = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'high_price_preference', 'Cluster']
            display_cols = [col for col in display_cols if col in st.session_state.df_features.columns]
            st.write(st.session_state.df_features[display_cols])
        else:
            default_clusters = min(4, n_samples)
            max_clusters = min(n_samples, 10)
            n_clusters = st.slider("Select number of clusters", min_value=1, max_value=max_clusters, value=default_clusters, key="n_clusters")
            basic_features = ['total_clicks', 'avg_price', 'unique_products', 'browsing_depth', 'price', 'high_price_preference']
            available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
            if len(available_features) == 0:
                st.error("No compatible features found for clustering. Ensure features like 'total_clicks', 'avg_price' are present.")
            else:
                try:
                    df_for_clustering = st.session_state.df_features[available_features]
                    scaler = preprocessor if preprocessor and isinstance(preprocessor, StandardScaler) else StandardScaler()
                    numeric_features = df_for_clustering.select_dtypes(include=['float64', 'int64']).columns
                    if len(numeric_features) == 0:
                        st.error("No numeric features available for clustering.")
                    else:
                        if not isinstance(preprocessor, StandardScaler) or not hasattr(scaler, 'mean_'):
                            scaler.fit(df_for_clustering[numeric_features])
                        df_for_clustering[numeric_features] = scaler.transform(df_for_clustering[numeric_features])
                        model = KMeans(n_clusters=n_clusters, random_state=42)
                        model.fit(df_for_clustering)
                        clusters = model.labels_
                        st.session_state.df_features["Cluster"] = clusters
                        st.write("### Clustering Results")
                        display_cols = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'high_price_preference', 'Cluster']
                        display_cols = [col for col in display_cols if col in st.session_state.df_features.columns]
                        st.write(st.session_state.df_features[display_cols])
                        st.write("### Cluster Centers (standardized features)")
                        cluster_centers_df = pd.DataFrame(model.cluster_centers_, columns=available_features)
                        st.write(cluster_centers_df)
                        st.write("### Cluster Visualizations")
                        cluster_counts = st.session_state.df_features["Cluster"].value_counts().reset_index()
                        cluster_counts.columns = ["Cluster", "Count"]
                        fig1 = px.pie(cluster_counts, names="Cluster", values="Count", title="Customer Segment Distribution")
                        st.plotly_chart(fig1)
                        fig2 = px.bar(cluster_counts, x="Cluster", y="Count", title="Number of Customers in Each Cluster",
                                      labels={'Count': 'Number of Customers'})
                        st.plotly_chart(fig2)
                        if 'total_clicks' in st.session_state.df_features.columns and 'avg_price' in st.session_state.df_features.columns:
                            fig3 = px.scatter(st.session_state.df_features, x='total_clicks', y='avg_price',
                                              color='Cluster', hover_data=['page1_main_category'] if 'page1_main_category' in st.session_state.df_features.columns else None,
                                              title="Cluster Distribution by Total Clicks and Average Price",
                                              labels={'total_clicks': 'Total Clicks', 'avg_price': 'Average Price ($)'})
                            st.plotly_chart(fig3)
                        st.write("### Cluster Characteristics")
                        numeric_cols = st.session_state.df_features.select_dtypes(include=['float64', 'int64']).columns
                        cluster_profiles = st.session_state.df_features.groupby('Cluster')[numeric_cols].mean()
                        st.write(cluster_profiles)
                except Exception as e:
                    st.error(f"Clustering failed: {e}")
                    st.write("Check your data or adjust the number of clusters.")
    else:
        st.warning("Please upload or manually enter data and ensure the clustering model is loaded.")

# Page: Help
elif page == "Help":
    st.title("Help & How to Access the Customer Conversion Analysis App")
    st.markdown("""
    This application helps you analyze customer clickstream data to predict purchase likelihood, spending amounts, and segment customers based on behavior. Below are instructions on how to access and use the app, along with tips for effective usage.

    ### Accessing the App
    The Customer Conversion Analysis app can be accessed in the following ways:

    #### 1. Running Locally
    To run the app on your local machine:
    - *Prerequisites*:
      - Install Python 3.8 or higher.
      - Install required packages by running:
        bash
        pip install streamlit pandas scikit-learn plotly numpy
        
      - Ensure the project directory contains:
        - app.py (this script)
        - Pickles/ folder with best_model_class.pkl, best_model_reg.pkl, best_model_clust.pkl, and preprocessed_data.pkl
        - Image/ folder with black-friday-elements-assortment.jpg
    - *Steps*:
      1. Navigate to the project directory:
         bash
         cd D:\\Projects\\Mini_Projects\\Clickstream_customer_conversion
         
      2. Run the Streamlit app:
         bash
         streamlit run app.py
         
      3. Open your web browser and go to http://localhost:8501 (or the URL displayed in the terminal).

    #### 2. Accessing a Deployed Version
    If the app is deployed (e.g., on Streamlit Community Cloud or another hosting service):
    - Open your web browser and navigate to the provided URL (e.g., https://your-app-name.streamlit.app).
    - Contact the app administrator for the exact URL if hosted externally.
    - No local installation is required; ensure you have a stable internet connection.

    #### 3. Accessing via Mobile Devices
    - If hosted online, the app can be accessed via web browsers on iOS or Android devices (e.g., Safari, Chrome).
    - Enter the deployed app’s URL in the browser.
    - Note: The app is not available as a native iOS or Android app but is mobile-responsive when accessed via a browser.

    ### Using the App
    The app has five main pages, accessible via the sidebar:
    - *Upload Data*: Upload a CSV file with clickstream data (expected columns: year, month, 'day`, session_id, order, country, etc.). Compute and save features for analysis.
    - *Manual Entry*: Enter individual data points manually using the form. Add multiple entries and compute features.
    - *Classification*: Predict whether customers will complete a purchase. Requires computed features from uploaded or manual data.
    - *Regression*: Predict customer spending amounts. Displays visualizations like scatter plots and histograms, and total prediction amount.
    - *Clustering*: Segment customers into groups based on behavior. Adjust the number of clusters using the slider.

    #### Tips for Effective Use
    - *Data Format*: Ensure CSV files include required columns (e.g., price, page2_clothing_model). Check the "Classification" or "Regression" debug info for missing features.
    - *Manual Entry*: Use the "Add Entry" button to input multiple records before computing features.
    - *Debugging*: Enable the "Show debug information" checkbox on analysis pages to troubleshoot feature mismatches.
    - *Saving Predictions*: Use the "Save Predictions" button to download results as a CSV file.
    - *Performance*: For large datasets, feature computation may take time. The app caches results to improve speed.

    ### Troubleshooting
    - *Model Loading Errors*: Verify that all .pkl files are in the Pickles/ folder and are not corrupted.
    - *Feature Mismatches*: If predictions fail, check debug info for missing features and ensure your data includes columns like total_clicks, avg_price, etc.
    - *Image Not Found*: The background image is optional. If missing, the app will proceed without it.
    - *Contact Support*: For issues, contact the app administrator or refer to the project documentation.

    For further assistance, revisit this Help page or check the project repository (if available).
    """)
    
    # =========================== Help & Footer =============================
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### How to use:
    
    1. *Upload your clickstream dataset* in CSV format, or use the *manual form input* option  
    2. Choose between *Classification, **Regression, or **Clustering* mode from the sidebar  
    3. Click *'Predict'* or *'Analyze'* to get customer conversion outcomes and insights  
    4. Go to the *Visualizations tab* to explore key plots like session durations, conversion funnels, and feature importance  
    5. Use the *Segment Insights tab* to understand customer behavior and segment-wise patterns  
""")

    st.sidebar.markdown("### About the model:")
    st.sidebar.markdown("""
    This application uses pre-trained Machine Learning models optimized for clickstream behavior analysis:
    - *Classification* models predict whether a user is likely to convert  
    - *Regression* models estimate *potential revenue* or *spend per session*  
    - *Clustering* groups users into behavioral segments (e.g., window shoppers, buyers, bounce users)

    Key features:
    - Data cleaning, feature engineering, and one-hot encoding pipelines  
    - Models include *Logistic Regression, **RandomForest, **KMeans*, and others  
    - Interactive prediction with *real-time feedback and visuals*
""")

    st.sidebar.markdown("### Evaluation metrics:")
    st.sidebar.markdown("""
    - *Classification:* Accuracy, Precision, Recall, F1-score, ROC-AUC  
    - *Regression:* MAE, RMSE, R² Score  
    - *Clustering:* Silhouette Score, Inertia, Segment Breakdown  
    - Visualization of confusion matrix, regression plots, and cluster assignments
""")

    st.sidebar.markdown("### Notes:")
    st.sidebar.markdown("""
    - No data is stored – *privacy is fully respected*  
    - Designed for *E-commerce and Retail Analytics* teams and researchers  
    - *App is in beta* – ongoing improvements and support for A/B testing metrics coming soon  
    - Use insights to optimize UX, marketing funnels, and conversion strategies  
    - *Happy analyzing – know your customers, grow your conversions! 📊*
    """)
if page in ["Classification", "Regression", "Clustering"]:
    if st.button("Save Predictions"):
        if "df_features" in st.session_state and st.session_state.df_features is not None:
            csv = st.session_state.df_features.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
            st.success("Predictions ready for download!")
        else:
            st.warning("No predictions available to save.")
