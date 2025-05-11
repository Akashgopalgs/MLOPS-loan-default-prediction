import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from feature_engineering import feature_engineering, save_feature_engineered_data


 
RAW_DATA_DIR = "C:/Users/akash/OneDrive/Documents/MLOPS/End_to_end_MLOPS_project/loan_default_mlops/data/raw"
PROCESSED_DATA_DIR = "C:/Users/akash/OneDrive/Documents/MLOPS/End_to_end_MLOPS_project/loan_default_mlops/data/processed"

def load_data(data_dir):
    files = [
        'application_train.csv', 'application_test.csv', 'bureau.csv',
        'bureau_balance.csv', 'POS_CASH_balance.csv', 'credit_card_balance.csv',
        'installments_payments.csv', 'previous_application.csv',
        'sample_submission.csv'
    ]
    data = {}
    for file in files:
        path = os.path.join(data_dir, file)
        data[file.split('.')[0]] = pd.read_csv(path)
    return data

def save_data(df, file_name):
    path = os.path.join(PROCESSED_DATA_DIR, file_name)
    df.to_csv(path, index=False)
    print(f"Saved cleaned file to: {path}")

def preprocess_application(df):
    # Drop columns with more than 40% missing values
    threshold = 0.4 * df.shape[0]
    df = df.dropna(thresh=threshold, axis=1)

    # Fill remaining NaNs
    for col in df.columns:
        if df[col].dtype == 'object':
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])
        else:
            imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
            df.loc[:, col] = imputer.fit_transform(df[[col]])

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df.loc[:, col] = le.fit_transform(df[col].astype(str))

    return df

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("Loading raw data...")
    data = load_data(RAW_DATA_DIR)
    train_df = data["application_train"]
    test_df = data["application_test"]

    print("Preprocessing training data...")
    train_df_cleaned = preprocess_application(train_df)

    print("Preprocessing test data...")
    test_df_cleaned = preprocess_application(test_df)

    print("Applying feature engineering to training data...")
    train_df_featured = feature_engineering(train_df_cleaned)

    print("Applying feature engineering to test data...")
    test_df_featured = feature_engineering(test_df_cleaned)

    save_feature_engineered_data(train_df_featured, "application_train_featured.csv")
    save_feature_engineered_data(test_df_featured, "application_test_featured.csv")

    print("✅ Preprocessing and feature engineering complete!")

if __name__ == "__main__":
    main()
