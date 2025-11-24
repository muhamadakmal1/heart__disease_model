import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = []
        
    def load_data(self):
        X_train = pd.read_csv('preprocessed_data/X_train.csv')
        X_val = pd.read_csv('preprocessed_data/X_val.csv')
        X_test = pd.read_csv('preprocessed_data/X_test.csv')
        y_train = pd.read_csv('preprocessed_data/y_train.csv').values.ravel()
        y_val = pd.read_csv('preprocessed_data/y_val.csv').values.ravel()
        y_test = pd.read_csv('preprocessed_data/y_test.csv').values.ravel()
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_ml_models(self, X_train, y_train, X_val, y_val):
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
        }
        
        print("Training ML models...")
        for name, model in models_config.items():
            print(f"  {name}...", end=' ')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            
            self.models[name] = model
            self.results.append({
                'Model': name,
                'Type': 'ML',
                'Accuracy': acc,
                'F1': f1,
                'AUC': auc
            })
            print(f"F1: {f1:.4f}")
    
    def tune_best_models(self, X_train, y_train):
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
            }
        }
        
        print("\nTuning models...")
        for name, params in param_grids.items():
            if name in self.models:
                print(f"  {name}...")
                search = RandomizedSearchCV(
                    self.models[name], params, n_iter=10, cv=3,
                    scoring='f1', n_jobs=-1, random_state=42
                )
                search.fit(X_train, y_train)
                self.models[f"{name}_tuned"] = search.best_estimator_
    
    def save_models(self):
        import os
        os.makedirs('models/ml_models', exist_ok=True)
        
        results_df = pd.DataFrame(self.results)
        ml_results = results_df[results_df['Type'] == 'ML']
        if len(ml_results) > 0:
            best_ml = ml_results.loc[ml_results['F1'].idxmax()]
            best_model = self.models[best_ml['Model']]
            joblib.dump(best_model, 'models/best_ml_model.pkl')
            print(f"\nBest ML: {best_ml['Model']} (F1: {best_ml['F1']:.4f})")
        
        results_df.to_csv('models/model_results.csv', index=False)
        print("\nModels saved successfully!")

def main():
    trainer = ModelTrainer()
    
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
    print(f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    
    trainer.train_ml_models(X_train, y_train, X_val, y_val)
    trainer.tune_best_models(X_train, y_train)
    trainer.save_models()

if __name__ == '__main__':
    main()
