"""
Test script for BPJS Fraud Detection API with RAG
"""
import requests
import json
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ API returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_prediction(csv_path: str, with_rag: bool = True):
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print(f"Testing Prediction Endpoint (RAG: {with_rag})")
    print("="*60)
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return None
    
    try:
        # Read and show preview
        df = pd.read_csv(csv_path)
        print(f"\n📁 File: {csv_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Send request
        with open(csv_path, 'rb') as f:
            files = {'file': (Path(csv_path).name, f, 'text/csv')}
            params = {'generate_rag_explanations': with_rag}
            
            print("\n🚀 Sending request...")
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                params=params,
                timeout=300
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Prediction successful!")
            print(f"   Total claims: {result.get('total_claims', 0)}")
            print(f"   Fraud detected: {result.get('fraud_detected', 0)}")
            
            if 'audit_reports_count' in result:
                print(f"   Audit reports: {result.get('audit_reports_count', 0)}")
            
            # Show sample prediction
            if result.get('predictions'):
                pred = result['predictions'][0]
                print("\n📊 Sample prediction:")
                print(f"   claim_id: {pred.get('claim_id', 'N/A')}")
                print(f"   predicted_fraud: {pred.get('predicted_fraud', 'N/A')}")
                print(f"   fraud_probability: {pred.get('fraud_probability', 0):.4f}")
                
                if pred.get('predicted_fraud') == 1:
                    print(f"   fraud_type: {pred.get('predicted_fraud_type', 'N/A')}")
                    print(f"   shap_summary: {pred.get('shap_explanation_summary', 'N/A')[:100]}...")
            
            # Show sample audit report
            if with_rag and result.get('audit_reports'):
                audit = result['audit_reports'][0]
                print("\n📋 Sample audit report:")
                print(f"   claim_id: {audit.get('claim_id', 'N/A')}")
                print(f"   fraud_type: {audit.get('fraud_type', 'N/A')}")
                print(f"   Explanation: {audit.get('explanation_text', 'N/A')[:200]}...")
                
                if audit.get('retrieved_docs'):
                    print(f"   Referenced docs: {len(audit['retrieved_docs'])}")
            
            return result
            
        else:
            print(f"\n❌ Prediction failed: {response.status_code}")
            print(response.text[:500])
            return None
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_explain_single(fraud_case: dict):
    """Test single explanation endpoint"""
    print("\n" + "="*60)
    print("Testing Single Explanation Endpoint")
    print("="*60)
    
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json=fraud_case,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Explanation generated!")
            print(f"   claim_id: {result.get('claim_id', 'N/A')}")
            print(f"   Explanation: {result.get('explanation_text', 'N/A')[:300]}...")
            return result
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text[:500])
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BPJS FRAUD DETECTION API - TEST SUITE")
    print("="*60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ API is not available. Please start the server first:")
        print("   uvicorn FastAPI.main:app --host 0.0.0.0 --port 8000")
        return
    
    # Test 2: Prediction WITHOUT RAG
    csv_path = input("\nEnter path to test CSV file: ").strip()
    if not csv_path:
        csv_path = "bpjs_inference_small.csv"  # Default
    
    print("\n--- Test: Prediction without RAG ---")
    result_no_rag = test_prediction(csv_path, with_rag=False)
    
    # Test 3: Prediction WITH RAG
    print("\n--- Test: Prediction with RAG ---")
    result_with_rag = test_prediction(csv_path, with_rag=True)
    
    # Test 4: Single explanation (if fraud cases found)
    if result_with_rag and result_with_rag.get('predictions'):
        fraud_cases = [
            p for p in result_with_rag['predictions'] 
            if p.get('predicted_fraud') == 1
        ]
        
        if fraud_cases:
            print("\n--- Test: Single case explanation ---")
            test_explain_single(fraud_cases[0])
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()