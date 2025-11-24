# Quick Reference

## Files

**Core Scripts:**

- `preprocess.py` - Clean and prepare data
- `train_models.py` - Train ML and DL models
- `app.py` - Flask web server
- `start.py` - Launch web app
- `run_pipeline.py` - Run full pipeline

**Analysis:**

- `1_exploratory_data_analysis.ipynb` - EDA notebook

**Web:**

- `templates/` - HTML pages
- `static/` - CSS and JavaScript

## Commands

```bash
# Full pipeline
python run_pipeline.py

# Just preprocess
python preprocess.py

# Just train models
python train_models.py

# Start web app
python start.py

# EDA notebook
jupyter notebook 1_exploratory_data_analysis.ipynb
```

## Generated Folders

After running scripts:

- `preprocessed_data/` - Train/val/test splits
- `models/` - Trained model files
- `eda_outputs/` - EDA visualizations

## Web Pages

- `/` - Home
- `/predict` - Make predictions
- `/dashboard` - View metrics
- `/about` - Project info

## Models Trained

**Machine Learning:**

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- SVM, KNN, Naive Bayes

Best models automatically saved to `models/`
