import pickle
from pathlib import Path

# Find latest binary model
artifacts_dir = Path('artifacts')
binary_dirs = sorted(artifacts_dir.glob('binary_ensemble_*'), reverse=True)

if binary_dirs:
    latest = binary_dirs[0]
    print(f"Checking: {latest}\n")
    
    # List all files
    print("Files in directory:")
    for file in latest.iterdir():
        print(f"  - {file.name}")
    
    # Try to load and check structure
    print("\nLoading artifacts...")
    
    from models.model_artifact_manager import ModelArtifactManager
    manager = ModelArtifactManager()
    
    artifacts = manager.load_artifacts(str(latest))
    
    print("\nArtifact keys:")
    for key in artifacts.keys():
        value = artifacts[key]
        if isinstance(value, dict):
            print(f"  - {key}: dict ({len(value)} items)")
            for subkey in list(value.keys())[:3]:
                print(f"      - {subkey}")
        else:
            print(f"  - {key}: {type(value).__name__}")
    
    # Check scaler specifically
    if 'scaler' in artifacts:
        print(f"\n✅ 'scaler' key found")
    else:
        print(f"\n❌ 'scaler' key NOT found")
        print(f"   Available keys: {list(artifacts.keys())}")
