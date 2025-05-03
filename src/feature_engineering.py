import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def feature_engineering(df):
    """
    Perform feature engineering on the dataset.
    Args:
        df (pd.DataFrame): Processed training or testing data.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    df = df.copy()  # Avoid modifying the original dataframe

    # Convert days to years
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'] / -365.0
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365.0

    # Create new ratio-based features
    df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    df['CHILDREN_RATIO'] = np.where(
        df['CNT_FAM_MEMBERS'] != 0,
        df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS'],
        0
    )

    df['TOTAL_PAYMENT'] = df['AMT_ANNUITY'] + df['AMT_GOODS_PRICE']

    df['AGE_EMPLOYED_RATIO'] = np.where(
        df['DAYS_EMPLOYED'] != 0,
        df['DAYS_BIRTH'] / df['DAYS_EMPLOYED'],
        0
    )

    df['LOG_AMT_INCOME'] = np.log1p(df['AMT_INCOME_TOTAL'])

    # Standardizing selected numeric features
    scaler = StandardScaler()
    features_to_scale = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    df['CREDIT_GOODS_RATIO'] = np.where(
        df['AMT_GOODS_PRICE'] != 0,
        df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'],
        0
    )
    joblib.dump(scaler, "models/scaler.pkl")
    return df


def save_feature_engineered_data(df, output_file_name):
    """
    Save the feature-engineered data to the processed folder.
    Args:
        df (pd.DataFrame): The feature-engineered dataset.
        output_file_name (str): The name of the file to save the data.
    """
    processed_data_path = f"data/processed/{output_file_name}"
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print(f"Feature-engineered data saved to: {processed_data_path}")

