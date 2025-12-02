"""
Full Pipeline: ML Inference → RAG Explanation
==============================================
Fixes path resolution issues between ML and RAG systems
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path


def run_full_pipeline(inference_data):
    """
    Run complete fraud detection pipeline:
    1. ML Inference (2-stage fraud detection)
    2. RAG Explanation (regulatory context)
    """
    
    # Setup paths
    ml_root = Path(__file__).parent.resolve()
    rag_dir = ml_root / "RAG"
    outputs_dir = ml_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = outputs_dir / f"predictions_{timestamp}.csv"
    rag_output_dir = rag_dir / "output"
    rag_output_dir.mkdir(exist_ok=True)
    rag_output_file = rag_output_dir / f"rag_explanations_{timestamp}.csv"
    
    print("\n" + "="*80)
    print("🚀 BPJS FRAUD DETECTION - FULL PIPELINE")
    print("="*80)
    print(f"📂 Working directory: {ml_root}")
    print(f"📂 RAG directory: {rag_dir}")
    print(f"📊 Input data: {inference_data}")
    print(f"💾 Predictions output: {predictions_file}")
    print(f"📝 RAG output: {rag_output_file}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: ML INFERENCE
    # ========================================================================
    print("\n🔮 Step 1: Running ML Inference...")
    print("-" * 80)
    
    ml_cmd = [
        "python", "main.py", "inference",
        "--data", inference_data,
        "--binary-model", "model_artifacts/binary_ensemble_20251120_194313",
        "--multiclass-model", "model_artifacts/multiclass_ensemble_20251120_194314",
        "--output", str(predictions_file)
    ]
    
    try:
        subprocess.run(ml_cmd, cwd=str(ml_root), check=True)
        print(f"\n✅ ML Inference completed.")
        print(f"   Predictions saved to: {predictions_file}")
        
        # Verify file exists
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not created: {predictions_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ML Inference failed with exit code {e.returncode}")
        return
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # ========================================================================
    # STEP 2: RAG EXPLANATION
    # ========================================================================
    print("\n📝 Step 2: Generating RAG Explanations...")
    print("-" * 80)
    
    # Use ABSOLUTE path for the input file
    predictions_abs_path = predictions_file.resolve()
    
    rag_cmd = [
        "python", "main.py", "explain",
        "--input", str(predictions_abs_path),  # ✅ ABSOLUTE PATH
        "--output", str(rag_output_file)
    ]
    
    try:
        subprocess.run(rag_cmd, cwd=str(rag_dir), check=True)
        print(f"\n✅ RAG explanations complete.")
        print(f"   Output saved to: {rag_output_file}")
        
        # Verify file exists
        if not rag_output_file.exists():
            raise FileNotFoundError(f"RAG output file not created: {rag_output_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ RAG explanation failed with exit code {e.returncode}")
        return
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 ML Predictions: {predictions_file}")
    print(f"📝 RAG Explanations: {rag_output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Configure your inference data path here
    inference_data = "bpjs_inference_small.csv"
    
    # Validate input file exists
    if not os.path.exists(inference_data):
        print(f"❌ Error: Input file not found: {inference_data}")
        print(f"   Current directory: {os.getcwd()}")
        exit(1)
    
    run_full_pipeline(inference_data)