import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(file_path):
    """
    Loads the dataset from the given file path.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
    return data

def aggregate_user_engagement(data):
    """
    Aggregates user engagement metrics including session count, average session duration, and total data volume.
    Args:
        data (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: Aggregated user engagement data.
    """
    user_metrics = data.groupby('MSISDN/Number').agg({
        'Dur. (ms)': ['count', 'mean', 'sum'],  
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()

   
    user_metrics.columns = ['User ID', 'Session Count', 'Avg Session Duration', 'Total Session Duration',
                            'Total DL (Bytes)', 'Total UL (Bytes)']

    user_metrics['Total Data Volume'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']

    print("\nAggregated User Engagement Metrics:")
    print(user_metrics.head())
    return user_metrics
def cluster_users(user_metrics, n_clusters=4):
    """
    Clusters users into engagement groups based on their metrics using K-Means.
    Args:
        user_metrics (pd.DataFrame): User engagement metrics.
        n_clusters (int): Number of clusters.
    Returns:
        pd.DataFrame: User metrics with cluster assignments.
    """
    features = ['Session Count', 'Avg Session Duration', 'Total Data Volume']
    X = user_metrics[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_metrics['Cluster'] = kmeans.fit_predict(X_scaled)

    print("\nCluster Centers:")
    print(kmeans.cluster_centers_)

    return user_metrics
def analyze_clusters(user_metrics):
    """
    Analyzes and visualizes the characteristics of each user cluster.
    Args:
        user_metrics (pd.DataFrame): User metrics with cluster assignments.
    """
    cluster_summary = user_metrics.groupby('Cluster').agg({
        'Session Count': 'mean',
        'Avg Session Duration': 'mean',
        'Total Data Volume': 'mean',
        'User ID': 'count'
    }).rename(columns={'User ID': 'User Count'})

    print("\nCluster Summary:")
    print(cluster_summary)

    
    cluster_summary[['Session Count', 'Avg Session Duration', 'Total Data Volume']].plot(kind='bar', figsize=(10, 6))
    plt.title("Cluster Characteristics")
    plt.xlabel("Cluster")
    plt.ylabel("Average Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def main():
    
    file_path = '../data/Copy of Week2_challenge_data_source(CSV).csv'

    
    data = load_data(file_path)
    user_metrics = aggregate_user_engagement(data)
    
    user_metrics = cluster_users(user_metrics)
    analyze_clusters(user_metrics)

   

if __name__ == "__main__":
    main()
