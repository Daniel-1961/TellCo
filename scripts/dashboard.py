import streamlit as st
import pandas as pd
from src.user_overview_analysis import (load_data, top_10_handsets, top_3_manufacturers, top_5_handsets_per_manufacturer, aggregate_application_data)
from src.user_engagement_analysis import (aggregate_user_engagement, cluster_users, analyze_clusters)
from src.experience_analytics import (aggregate_customer_data, perform_clustering)
from src.satisfaction_analysis import (calculate_scores, calculate_satisfaction_score, kmeans_clustering)

# Load data
@st.cache
def load_raw_data():
    return pd.read_csv('data/Copy of Week2_challenge_data_source(CSV).csv')

data = load_raw_data()

# Sidebar navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", [
    "User Overview Analysis", "User Engagement Analysis",
    "Experience Analysis", "Satisfaction Analysis"
])

if page == "User Overview Analysis":
    st.title("User Overview Analysis")

    # Display top 10 handsets
    st.subheader("Top 10 Handsets")
    top_handsets = top_10_handsets(data)
    st.bar_chart(top_handsets)

    # Display top 3 manufacturers
    st.subheader("Top 3 Manufacturers")
    top_manufacturers = top_3_manufacturers(data)
    st.write(top_manufacturers)

    # Display top 5 handsets per manufacturer
    st.subheader("Top 5 Handsets Per Manufacturer")
    if top_manufacturers is not None:
        top_5_handsets_per_manufacturer(data, top_manufacturers)

    # Aggregated application data
    st.subheader("Aggregated Application Data")
    aggregated_data = aggregate_application_data(data)
    st.write(aggregated_data)


# User Engagement Analysis Page
elif page == "User Engagement Analysis":
    st.title("User Engagement Analysis")
    
    # Aggregate user engagement data
    st.subheader("Aggregated User Engagement Metrics")
    user_metrics = aggregate_user_engagement(data)
    st.write(user_metrics)
    
    # Cluster users
    st.subheader("User Clustering")
    clustered_data = cluster_users(user_metrics)
    st.write(clustered_data)
    
    # Analyze clusters
    st.subheader("Cluster Analysis")
    analyze_clusters(clustered_data)

# Experience Analysis Page
elif page == "Experience Analysis":
    st.title("Experience Analysis")
    
    # Aggregate customer data
    st.subheader("Aggregated Customer Data")
    customer_data = aggregate_customer_data(data)
    st.write(customer_data)
    
    # Perform clustering
    st.subheader("Experience Clustering")
    clustered_customer_data = perform_clustering(customer_data)
    st.write(clustered_customer_data)

# Satisfaction Analysis Page
elif page == "Satisfaction Analysis":
    st.title("Satisfaction Analysis")
 
    # Engagement and experience scores
    st.subheader("Engagement and Experience Scores")
    engagement_clusters = [
    [3.367763e+10, 1.959627e+07, 1.041968e+06],  # Cluster 0: Moderate Engagement
    [4.749426e+10, 1.625573e+07, 6.719899e+05],  # Cluster 1: Low Engagement (worst)
    [3.368364e+10, 2.224330e+09, 1.161338e+06],  # Cluster 2: High Engagement (best)
    ]
    experience_clusters = [
    [77.319787, 24.225670, 49171.539659],  # Cluster 0: Moderate Experience
    [126.372621, 15.220035, 3799.595727],  # Cluster 1: Low Experience (worst)
    [104.547799, 39.244130, 62838.408386],  # Cluster 2: High Experience (best)
    ]
    scored_data = calculate_scores(data, engagement_clusters, experience_clusters)
    scored_data = calculate_satisfaction_score(scored_data)
    st.write(scored_data)
    
    # K-means clustering on satisfaction data
    st.subheader("Satisfaction Clustering")
    clustered_data = kmeans_clustering(scored_data)
    st.write(clustered_data)
