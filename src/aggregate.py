import pandas as pd

def aggregate_user_behavior(data):
    """
    Aggregate per user the behavior metrics.
    :param data: DataFrame containing user session data.
    :return: Aggregated DataFrame.
    """
    # Ensure relevant columns exist in the dataset
    required_columns = ['Bearer Id ', 'Dur. (ms)', 'Total UL', 'Total DL']
    
    # Calculate total data volume if not already in the dataset
    
    data['Total_Data_Volume'] = data['Total UL'] + data['Total DL']

    # Aggregate data per user
    aggregated_data = data.groupby('Bearer Id').agg(
        xDR_sessions=('Bearer  Id', 'count'),  
        total_session_duration=('Dur. (ms)', 'sum'), 
        total_download_data=('Total DL', 'sum'),  
        total_upload_data=('Total UP', 'sum'),  
        total_data_volume=('Total_Data_Volume', 'sum')
    ).reset_index()

    return aggregated_data

#Data destination
output_file = "../Data/processed_data.csv" 

# Load data
df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\Copy of Week2_challenge_data_source(CSV).csv")

# Perform aggregation
aggregated_data = aggregate_user_behavior(df)

# Save the aggregated data
aggregated_data.to_csv(output_file, index=False)
print(f"Aggregated user data saved to {output_file}")
