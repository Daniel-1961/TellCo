import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine

def calculate_scores(data, engagement_clusters, experience_clusters):
    """
    Calculate engagement and experience scores based on Euclidean distance to clusters.
    Args:
        data (pd.DataFrame): The dataset.
        engagement_clusters (ndarray): Cluster centers from engagement analysis.
        experience_clusters (ndarray): Cluster centers from experience analysis.
    Returns:
        pd.DataFrame: Data with engagement and experience scores.
    """
    
    data['Session Count'] = data.groupby('MSISDN/Number')['Bearer Id'].transform('count')
    data['Avg Session Duration'] = data.groupby('MSISDN/Number')['Dur. (ms)'].transform('mean')
    data['Total Data Volume'] = data.groupby('MSISDN/Number')['Total DL (Bytes)'].transform('sum') + \
                                data.groupby('MSISDN/Number')['Total UL (Bytes)'].transform('sum')

    engagement_features = ['Session Count', 'Avg Session Duration', 'Total Data Volume']

   
    data['Engagement Score'] = data[engagement_features].apply(
        lambda row: np.linalg.norm(row - engagement_clusters[1]), axis=1)

  
    data['Avg TCP Retransmission'] = (data['TCP DL Retrans. Vol (Bytes)'] + data['TCP UL Retrans. Vol (Bytes)']) / 2
    data['Avg RTT DL'] = data['Avg RTT DL (ms)']
    data['Avg Throughput DL'] = data['Avg Bearer TP DL (kbps)']

    experience_features = ['Avg TCP Retransmission', 'Avg RTT DL', 'Avg Throughput DL']

   
    data['Experience Score'] = data[experience_features].apply(
        lambda row: np.linalg.norm(row - experience_clusters[1]), axis=1)

    return data
def calculate_satisfaction_score(data):
    """
    Calculate the satisfaction score as the average of engagement and experience scores.
    Args:
        data (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: Data with satisfaction scores.
    """
  
    data['Satisfaction Score'] = (data['Engagement Score'] + data['Experience Score']) / 2

   
    top_customers = data.nlargest(10, 'Satisfaction Score')[['MSISDN/Number', 'Satisfaction Score']]
    print("\nTop 10 Satisfied Customers:")
    print(top_customers)

    return data

def build_regression_model(data):
    """
    Build a regression model to predict satisfaction scores.
    Args:
        data (pd.DataFrame): The dataset.
    """
   
    X = data[['Engagement Score', 'Experience Score']]
    y = data['Satisfaction Score']

    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

   
    y = y.dropna()
    X = X[:len(y)]  

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LinearRegression()
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRegression Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    return model


def kmeans_clustering(data):
    """
    Perform K-means clustering on engagement and experience scores.
    Args:
        data (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: Data with cluster assignments.
    """
    
    features = ['Engagement Score', 'Experience Score']
    X = data[features].dropna()  

    
    kmeans = KMeans(n_clusters=2, random_state=42)
    data.loc[X.index, 'Cluster'] = kmeans.fit_predict(X)

    
    plt.figure(figsize=(10, 6))
    plt.scatter(X['Engagement Score'], X['Experience Score'], c=kmeans.labels_, cmap='viridis', alpha=0.7)
    plt.title("K-Means Clustering (k=2)")
    plt.xlabel("Engagement Score")
    plt.ylabel("Experience Score")
    plt.colorbar(label="Cluster")
    plt.show()

    return data
def aggregate_cluster_scores(data):
    """
    Aggregate the average satisfaction and experience scores per cluster.
    Args:
        data (pd.DataFrame): The dataset.
    """
    
    cluster_data = data.dropna(subset=['Cluster', 'Satisfaction Score', 'Experience Score'])

   
    cluster_summary = cluster_data.groupby('Cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean'
    })

    print("\nCluster Summary:")
    print(cluster_summary)

    return cluster_summary


def export_to_postgres(data, db_url, table_name):
    """
    Export data to a PostgreSQL database.
    Args:
        data (pd.DataFrame): The dataset.
        db_url (str): Database connection string.
        table_name (str): Name of the table.
    """
    
    export_data = data[['MSISDN/Number', 'Engagement Score', 'Experience Score', 'Satisfaction Score', 'Cluster']]
    export_data = export_data.dropna()  

   
    engine = create_engine(db_url)

    
    export_data.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"\nData exported to PostgreSQL table '{table_name}'")

def main():
    
    file_path = '../data/Copy of Week2_challenge_data_source(CSV).csv'
    data = pd.read_csv(file_path)

   
    engagement_clusters = np.array([
    [3.367763e+10, 1.959627e+07, 1.041968e+06],  # Cluster 0: Moderate Engagement
    [4.749426e+10, 1.625573e+07, 6.719899e+05],  # Cluster 1: Low Engagement (worst)
    [3.368364e+10, 2.224330e+09, 1.161338e+06],  # Cluster 2: High Engagement (best)
    ])

    experience_clusters = np.array([
    [77.319787, 24.225670, 49171.539659],  # Cluster 0: Moderate Experience
    [126.372621, 15.220035, 3799.595727],  # Cluster 1: Low Experience (worst)
    [104.547799, 39.244130, 62838.408386],  # Cluster 2: High Experience (best)
    ])

    # Calculate engagement features
    data['Avg TCP Retransmission'] = (data['TCP DL Retrans. Vol (Bytes)'] + data['TCP UL Retrans. Vol (Bytes)']) / 2
    data['Avg RTT DL'] = data['Avg RTT DL (ms)']
    data['Avg Throughput DL'] = data['Avg Bearer TP DL (kbps)']


    # Calculate session count
    data['Session Count'] = data.groupby('MSISDN/Number')['Bearer Id'].transform('count')
    data['Avg Session Duration'] = data.groupby('MSISDN/Number')['Dur. (ms)'].transform('mean')
    data['Total Data Volume'] = data.groupby('MSISDN/Number')['Total DL (Bytes)'].transform('sum') + \
                            data.groupby('MSISDN/Number')['Total UL (Bytes)'].transform('sum')
   
    data = calculate_scores(data, engagement_clusters, experience_clusters)
    data = calculate_satisfaction_score(data)

    build_regression_model(data)
    data = kmeans_clustering(data)
    cluster_summary = aggregate_cluster_scores(data)
    db_url = 'postgresql+psycopg2://hunegna_2:12372342@localhost:5432/telecom_analysis'
    export_to_postgres(data, db_url, 'satisfaction_analysis')


   

if __name__ == "__main__":
    main()
