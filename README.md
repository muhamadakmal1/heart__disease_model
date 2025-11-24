# Heart Disease Prediction

Machine learning web app for predicting heart disease risk.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train models
```bash
python preprocess.py
python train_models.py
```

### Run web app
```bash
python start.py
```

Visit http://localhost:5000

### EDA notebook
```bash
jupyter notebook 1_exploratory_data_analysis.ipynb
```

## Project Structure

```
├── heart_disease.csv          # dataset
├── preprocess.py              # data preprocessing  
├── train_models.py            # model training
├── app.py                     # web server
├── start.py                   # launcher
├── templates/                 # html pages
├── static/                    # css & js
└── 1_exploratory_data_analysis.ipynb  # eda notebook
```

## Models

- Machine Learning: Random Forest, XGBoost, LightGBM, etc.
- Deep Learning: Neural network with dropout
- Expected accuracy: 90-92%

## Features

- Interactive web interface
- ML vs DL model selection
- Real-time predictions
- Dashboard with metrics
- Responsive design

## Tech Stack

- Backend: Flask
- ML: scikit-learn, XGBoost, LightGBM
- DL: TensorFlow/Keras
- Frontend: HTML, CSS, JavaScript, Chart.js

## License

MIT
