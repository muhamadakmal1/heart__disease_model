import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean_data():
    df = pd.read_csv('heart_disease.csv')
    print(f"Loaded data: {df.shape}")
    
    # handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    target = 'Heart Disease Status'
    categorical_cols.remove(target)
    
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df, numerical_cols, categorical_cols, target, num_imputer, cat_imputer

def engineer_features(df):
    # create age groups
    df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], 
                             labels=['young', 'middle', 'senior', 'elderly'])
    
    # BMI categories
    df['bmi_cat'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                           labels=['underweight', 'normal', 'overweight', 'obese'])
    
    # risk factors
    df['smoker'] = (df['Smoking'] == 'Yes').astype(int)
    df['has_diabetes'] = (df['Diabetes'] == 'Yes').astype(int)
    df['high_bp'] = (df['High Blood Pressure'] == 'Yes').astype(int)
    df['family_history'] = (df['Family Heart Disease'] == 'Yes').astype(int)
    
    # risk score
    df['risk_score'] = df['smoker'] + df['has_diabetes'] + df['high_bp'] + df['family_history']
    
    return df

def prepare_data(df, target):
    # encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    
    # drop original target and get features
    X = df.drop(target, axis=1)
    
    # one-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    
    # scale features
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler, le

def split_data(X, y):
    # 70-15-15 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler, le):
    os.makedirs('preprocessed_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train.to_csv('preprocessed_data/X_train.csv', index=False)
    X_val.to_csv('preprocessed_data/X_val.csv', index=False)
    X_test.to_csv('preprocessed_data/X_test.csv', index=False)
    
    pd.Series(y_train).to_csv('preprocessed_data/y_train.csv', index=False, header=['target'])
    pd.Series(y_val).to_csv('preprocessed_data/y_val.csv', index=False, header=['target'])
    pd.Series(y_test).to_csv('preprocessed_data/y_test.csv', index=False, header=['target'])
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/target_encoder.pkl')
    
    pd.Series(X_train.columns).to_csv('preprocessed_data/feature_names.csv', 
                                       index=False, header=['feature'])
    
    print("Data saved successfully!")

def main():
    print("Starting preprocessing...")
    
    df, num_cols, cat_cols, target, num_imp, cat_imp = load_and_clean_data()
    df = engineer_features(df)
    X, y, scaler, le = prepare_data(df, target)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    save_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler, le)
    
    print(f"\nPreprocessing complete!")
    print(f"Features: {X.shape[1]}")

if __name__ == '__main__':
    main()
