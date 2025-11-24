import subprocess
import sys

def run_step(script, description):
    print(f"\n{'='*50}")
    print(f"{description}")
    print('='*50)
    
    try:
        subprocess.run([sys.executable, script], check=True)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("Heart Disease Prediction Pipeline")
    print("="*50)
    
    steps = [
        ('preprocess.py', 'Step 1: Preprocessing Data'),
        ('train_models.py', 'Step 2: Training Models'),
    ]
    
    for script, desc in steps:
        if not run_step(script, desc):
            print(f"\nFailed at: {desc}")
            break
    
    print("\n" + "="*50)
    print("Pipeline Complete!")
    print("="*50)
    print("\nNext steps:")
    print("  - Check 'models/' for trained models")
    print("  - Run 'python app.py' to start web app")

if __name__ == '__main__':
    main()
