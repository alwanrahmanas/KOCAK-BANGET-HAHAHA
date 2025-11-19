from pathlib import Path
import joblib

# Find latest binary model
artifacts_dir = Path('artifacts')
binary_dirs = sorted(artifacts_dir.glob('binary_ensemble_*'), reverse=True)

if binary_dirs:
    latest = binary_dirs[0]
    print(f"Checking: {latest}\n")
    
    # List all files
    print("Files:")
    for file in latest.iterdir():
        print(f"  - {file.name}")
    
    # Check if scaler.pkl exists
    scaler_path = latest / 'scaler.pkl'
    if scaler_path.exists():
        print(f"\n✅ scaler.pkl EXISTS")
        
        # Try to load it
        try:
            scaler = joblib.load(scaler_path)
            print(f"   Type: {type(scaler)}")
            print(f"   Attributes: {dir(scaler)}")
        except Exception as e:
            print(f"   ❌ Error loading: {e}")
    else:
        print(f"\n❌ scaler.pkl DOES NOT EXIST!")
        print(f"   This is why inference fails.")
        print(f"   You need to re-train the model.")
