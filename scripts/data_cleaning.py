import pandas as pd
import numpy as np
def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pd.read_csv(file_path)

def handle_missing_values(data):
    """Handles missing values by replacing with mean (numerical) or mode (categorical)."""
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
    return data

def treat_outliers_iqr(data, numerical_cols):
    """Treats outliers in numerical columns using the IQR method."""
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = np.where(data[col] < lower_bound, data[col].mean(), data[col])
        data[col] = np.where(data[col] > upper_bound, data[col].mean(), data[col])
    return data

def save_cleaned_data(data, output_path):
    """Save cleaned data to a specified file path."""
    data.to_csv(output_path, index=False)
if __name__ == "__main__":
#     #Load the raw dataset
     file_path ="C:\\Users\\lenovo\\OneDrive\\Desktop\\Copy of Week2_challenge_data_source(CSV).csv"  
     data = load_data(file_path)

#     # Handle missing values
data = handle_missing_values(data)

    # Treat outliers
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
data=treat_outliers_iqr(data, numerical_cols)

    # Save cleaned dataset
cleaned_file_path="Daniel-1961\\TellCo\\Data\\cleaned_data.csv" 
save_cleaned_data(data, cleaned_file_path)
print(f"Cleaned data saved to {cleaned_file_path}")
