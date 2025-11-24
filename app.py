from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'heart-prediction-2025'

ml_model = None

def load_models():
    global ml_model
    try:
        if os.path.exists('models/best_ml_model.pkl'):
            ml_model = joblib.load('models/best_ml_model.pkl')
            print("ML model loaded")
    except Exception as e:
        print(f"Error: {e}")

load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        X_test = pd.read_csv('preprocessed_data/X_test.csv')
        sample = X_test.iloc[np.random.randint(0, len(X_test))].values.reshape(1, -1)
        
        if ml_model:
            pred = ml_model.predict(sample)[0]
            proba = ml_model.predict_proba(sample)[0] if hasattr(ml_model, 'predict_proba') else [0.5, 0.5]
            
            return jsonify({
                'prediction': 'Heart Disease' if pred == 1 else 'No Heart Disease',
                'probability': {
                    'disease': float(proba[1] * 100),
                    'no_disease': float(proba[0] * 100)
                },
                'confidence': float(max(proba) * 100),
                'model': 'Machine Learning'
            })
        
        return jsonify({'error': 'Model not available'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/stats')
def stats():
    try:
        X_train = pd.read_csv('preprocessed_data/X_train.csv')
        return jsonify({
            'total_records': 10000,
            'total_features': len(X_train.columns),
            'train_size': len(X_train),
            'val_size': 1496,
            'test_size': 1500
        })
    except:
        return jsonify({'error': 'Stats not available'}), 404

if __name__ == '__main__':
    print("="*60)
    print("Heart Disease Prediction - Web App")
    print("="*60)
    print(f"ML Model: {'Loaded' if ml_model else 'Not loaded'}")
    print(f"\nOpen: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

