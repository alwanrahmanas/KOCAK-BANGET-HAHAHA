"""
Simple test to verify API is working
"""
import requests
import json

API_URL = "http://localhost:8000"

def test():
    print("="*60)
    print("QUICK API TEST")
    print("="*60)
    
    # 1. Health check
    print("\n[1] Testing health endpoint...")
    try:
        r = requests.get(f"{API_URL}/health")
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    # 2. Test with sample data
    print("\n[2] Testing prediction endpoint...")
    
    # Create minimal CSV in memory
    import io
    csv_data = """claim_id,age,billed_amount,paid_amount,lama_dirawat,tarif_inacbg,claim_ratio,sex,faskes_level,jenis_pelayanan,room_class,kapitasi_flag,referral_flag,referral_to_same_facility,visit_suspicious_flag,obat_mismatch_flag,billing_spike_flag,high_deviation_flag,visit_count_30d,provider_monthly_claims,drug_cost,procedure_cost,drug_ratio,procedure_ratio,time_diff_prev_claim,rolling_avg_cost_30d,nik_hash_reuse_count,provider_claim_share,claim_month,claim_quarter,kode_icd10,diagnosis_name,clinical_pathway_name,kode_prosedur,procedure_name,inacbg_code,kode_tarif_inacbg,faskes_id,selisih_klaim,clinical_pathway_deviation_score,total_klaim_5x,rerata_billed_5x,std_claim_ratio_5x,rerata_lama_dirawat_5x,total_rs_unique_visited_5x,total_diagnosis_unique_5x,obat_match_score,phantom_suspect_score,phantom_node_flag
CLM00001,45,5000000,4500000,3,4000000,1.25,L,Primer,Rawat Inap,Kelas 3,0,1,0,0,0,0,0,2,150,1500000,2000000,0.3,0.4,15.5,4200000,1,0.05,6,2,A12,Diabetes,DM Type 2,12.34,Lab Test,A-1-23-I,INV1234,FASK001,500000,0.15,3,4800000,0.12,3.5,2,4,0.85,0.1,0
CLM00002,67,8000000,7500000,7,6000000,1.33,F,Sekunder,Rawat Inap,Kelas 2,0,1,0,0,1,1,1,5,280,3000000,3500000,0.375,0.4375,8.2,6500000,1,0.08,8,3,B45,Pneumonia,CAP,45.67,Radiologi,B-2-45-I,INV5678,FASK002,500000,0.65,5,7200000,0.18,5.2,3,6,0.72,0.25,0
CLM00003,34,3500000,3200000,2,3000000,1.17,M,Primer,Rawat Jalan,Kelas 3,1,0,0,0,0,0,0,1,120,800000,1200000,0.229,0.343,45.8,3100000,1,0.03,4,2,C78,Gastritis,Acute Gastritis,78.90,Medication,C-3-78-I,INV9012,FASK003,300000,0.08,2,3400000,0.09,2.8,1,3,0.91,0.05,0"""
    
    try:
        files = {
            'file': ('test.csv', io.StringIO(csv_data), 'text/csv')
        }
        
        print("Sending 3 test claims...")
        r = requests.post(
            f"{API_URL}/predict",
            files=files,
            params={'generate_rag_explanations': False},  # Fast test
            timeout=60
        )
        
        print(f"Status: {r.status_code}")
        
        if r.status_code == 200:
            result = r.json()
            print(f"✅ Success!")
            print(f"   Total claims: {result.get('total_claims')}")
            print(f"   Fraud detected: {result.get('fraud_detected')}")
            
            if result.get('predictions'):
                print(f"\n   Sample prediction:")
                pred = result['predictions'][0]
                print(f"   - claim_id: {pred.get('claim_id')}")
                print(f"   - fraud: {pred.get('predicted_fraud')}")
                print(f"   - probability: {pred.get('fraud_probability', 0):.4f}")
        else:
            print(f"❌ Failed")
            print(r.text[:500])
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()