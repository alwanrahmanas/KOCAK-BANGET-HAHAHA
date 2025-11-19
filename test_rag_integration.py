"""
Test RAG Integration
Quick test to verify RAG system connection
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from rag_integration import RAGIntegration
import pandas as pd

def test_rag_integration():
    """Test RAG integration"""
    
    print("\n" + "="*80)
    print("TESTING RAG INTEGRATION")
    print("="*80)
    
    # Test 1: Check availability
    print("\n[Test 1] Checking RAG availability...")
    rag = RAGIntegration()
    
    status = rag.get_status()
    print(f"  Available: {status['available']}")
    print(f"  Path: {status['path']}")
    print(f"  Initialized: {status['initialized']}")
    
    if not rag.is_available():
        print("\n❌ RAG system not available")
        print("   Please check:")
        print("   1. RAG_SYSTEM_PATH in .env")
        print("   2. RAG system files exist")
        return False
    
    print("  ✓ RAG system available")
    
    # Test 2: Initialize RAG
    print("\n[Test 2] Initializing RAG system...")
    try:
        rag.initialize_rag_system(rebuild_vectors=False)
        print("  ✓ RAG system initialized")
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return False
    
    # Test 3: Test with sample fraud case
    print("\n[Test 3] Testing with sample fraud case...")
    
    # Create sample predictions
    sample_predictions = pd.DataFrame([{
        'claim_id': 'TEST_001',
        'predicted_fraud': 1,
        'fraud_probability': 0.95,
        'predicted_fraud_type': 'upcoding_diagnosis',
        'shap_explanation_summary': 'claim_ratio↑, selisih_klaim↑',
    }])
    
    # Create sample original claims
    sample_claims = pd.DataFrame([{
        'claim_id': 'TEST_001',
        'kode_icd10': 'A09.0',
        'kode_prosedur': '87.03',
        'jenis_pelayanan': 'Rawat Inap',
        'room_class': 'Kelas 1',
        'lama_dirawat': 1,
        'billed_amount': 8000000,
        'tarif_inacbg': 5000000,
        'selisih_klaim': 3000000,
        'claim_ratio': 1.6,
    }])
    
    try:
        result_df = rag.generate_explanations(
            predictions_df=sample_predictions,
            original_claims_df=sample_claims,
            fraud_only=True
        )
        
        print(f"  ✓ Generated explanation")
        print(f"  Columns: {list(result_df.columns)}")
        
        if 'explanation_text' in result_df.columns:
            explanation = result_df.iloc[0]['explanation_text']
            print(f"\n  Sample explanation (first 300 chars):")
            print(f"  {explanation[:300]}...")
            
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Explanation generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_rag_integration()
    sys.exit(0 if success else 1)
