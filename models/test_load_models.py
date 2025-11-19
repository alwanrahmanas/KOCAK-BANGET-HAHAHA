"""
Test script to verify ModelArtifactManager methods
"""

import sys
from pathlib import Path

# Add parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now import will work
from models.model_artifact_manager import ModelArtifactManager

# Test if method exists
manager = ModelArtifactManager()

print("="*60)
print("CHECKING MODELARTIFACTMANAGER METHODS")
print("="*60)

print("\nChecking methods:")
print(f"  Has 'load_artifacts':          {hasattr(manager, 'load_artifacts')}")
print(f"  Has 'load_pipeline_artifacts': {hasattr(manager, 'load_pipeline_artifacts')}")
print(f"  Has 'save_pipeline':           {hasattr(manager, 'save_pipeline')}")

# List all public methods
methods = [m for m in dir(manager) if not m.startswith('_') and callable(getattr(manager, m))]
print(f"\nAll available methods:")
for method in methods:
    print(f"  - {method}")

# Try loading test
print("\n" + "="*60)
print("TESTING ARTIFACT LOADING")
print("="*60)

# Check if binary model directory exists
import os
artifacts_dir = Path(parent_dir) / 'artifacts'

if artifacts_dir.exists():
    # Find first binary model directory
    binary_dirs = list(artifacts_dir.glob('binary_ensemble_*'))
    
    if binary_dirs:
        test_dir = binary_dirs[0]
        print(f"\nTesting with: {test_dir.name}")
        
        try:
            if hasattr(manager, 'load_artifacts'):
                artifacts = manager.load_artifacts(str(test_dir))
                print("\n✅ load_artifacts() works!")
                print(f"   Loaded keys: {list(artifacts.keys())}")
            elif hasattr(manager, 'load_pipeline_artifacts'):
                artifacts = manager.load_pipeline_artifacts(str(test_dir))
                print("\n✅ load_pipeline_artifacts() works!")
                print(f"   Loaded keys: {list(artifacts.keys())}")
            else:
                print("\n❌ No load method found!")
        except Exception as e:
            print(f"\n❌ Error loading artifacts: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No binary model directories found in artifacts/")
        print("   Run training first: python main.py train")
else:
    print(f"\n⚠️  Artifacts directory not found: {artifacts_dir}")
    print("   Run training first: python main.py train")
