import subprocess
import sys
import os

def check_setup():
    if not os.path.exists('preprocessed_data/X_train.csv'):
        print("Data not preprocessed yet!")
        print("\nRun: python preprocess.py")
        return False
    
    if not os.path.exists('models/best_ml_model.pkl'):
        print("Models not trained yet!")
        print("\nRun: python train_models.py")
        return False
    
    return True

def main():
    print("Launching web app...")
    
    if check_setup():
        print("Starting server...\n")
        subprocess.run([sys.executable, 'app.py'])
    else:
        print("\nSetup required first")

if __name__ == '__main__':
    main()
