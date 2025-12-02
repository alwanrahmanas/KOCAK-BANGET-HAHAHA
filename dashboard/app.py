import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO, BytesIO
from datetime import datetime


import threading
import uvicorn

def start_api():
    uvicorn.run("FastAPI.app:app", host="0.0.0.0", port=8000)

if "api_started" not in st.session_state:
    threading.Thread(target=start_api, daemon=True).start()
    st.session_state["api_started"] = True

# ==================== CONFIG ====================
API_URL = "http://0.0.0.0:8000" # Cloud Server

st.set_page_config(
    page_title="BPJS Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

# ==================== HELPER FUNCTIONS ====================
def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def run_inference(uploaded_file, generate_rag=True):
    """Send file to API for inference"""
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        files = {
            "file": (uploaded_file.name, uploaded_file, "text/csv")
        }
        
        # Add query parameter for RAG generation
        params = {"generate_rag_explanations": generate_rag}
        
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            params=params,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return False, "Request timeout (>5 minutes)"
    except Exception as e:
        return False, str(e)

def create_csv_from_dict(data_dict):
    """Convert single claim dict to CSV BytesIO for API"""
    df = pd.DataFrame([data_dict])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer

# ==================== MAIN DASHBOARD ====================
st.title("🏥 BPJS Fraud Detection Dashboard")
st.markdown("---")

# ========== Sidebar ==========
with st.sidebar:
    st.header("System Status")
    
    # API Health Check
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("✅ API Connected")
        if isinstance(health_data, dict):
            st.json(health_data)
    else:
        st.error("❌ API Unavailable")
        st.code(health_data)
        st.warning("Please start the FastAPI server:\n``````")
    
    st.markdown("---")
    st.info("""
    **How to use:**
    
    **Batch Mode (CSV):**
    1. Upload CSV file
    2. Preview data
    3. Click 'Run Inference'
    
    **Single Claim Mode:**
    1. Fill in claim details
    2. Click 'Analyze Claim'
    3. View fraud prediction
    """)

# ========== Input Mode Selection ==========
st.markdown("### 📥 Input Mode")
input_mode = st.radio(
    "Choose input method:",
    ["📁 Upload CSV (Batch)", "✍️ Single Claim Input"],
    horizontal=True
)
st.markdown("---")

# ==================== MODE 1: CSV UPLOAD (BATCH) ====================
if input_mode == "📁 Upload CSV (Batch)":
    
    uploaded_file = st.file_uploader(
        "📁 Upload CSV File", 
        type=["csv"],
        help="Upload a CSV file containing claim data for fraud detection"
    )
    
    if uploaded_file is not None:
        # Preview uploaded data
        st.subheader("📊 Data Preview")
        
        try:
            df_preview = pd.read_csv(uploaded_file)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df_preview))
            col2.metric("Total Columns", len(df_preview.columns))
            col3.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
            
            st.dataframe(df_preview.head(10), use_container_width=True)
            
            # Show column info
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df_preview.columns,
                    'Type': df_preview.dtypes.astype(str), 
                    'Non-Null': df_preview.count().values,
                    'Null': df_preview.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to preview file: {e}")
        
        # Run Inference Button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            btn_run_inference = st.button(
                "🚀 Run Fraud Detection", 
                type="primary", 
                use_container_width=True
            )
        
        with col2:
            generate_rag = st.checkbox(
                "Generate RAG Reports", 
                value=True,
                help="Generate detailed audit reports using RAG system"
            )
        
        if btn_run_inference:
            
            if not is_healthy:
                st.error("❌ Cannot run inference: API is not available")
                st.stop()
            
            with st.spinner("🔄 Running fraud detection... This may take a few minutes..."):
                
                api_success, api_result = run_inference(uploaded_file, generate_rag)
                
                if api_success:
                    st.success("✅ Inference completed successfully!")
                    
                    # Extract results
                    if isinstance(api_result, dict):
                        # Parse response structure
                        total_claims = api_result.get("total_claims", 0)
                        fraud_detected = api_result.get("fraud_detected", 0)
                        predictions = api_result.get("predictions", [])
                        
                        if predictions:
                            result_df = pd.DataFrame(predictions)
                            
                            # ========== Summary Metrics ==========
                            st.subheader("📈 Detection Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric(
                                "Total Claims",
                                total_claims
                            )
                            
                            col2.metric(
                                "Fraud Detected",
                                fraud_detected,
                                delta=f"{(fraud_detected/total_claims*100):.1f}%" if total_claims > 0 else "0%"
                            )
                            
                            col3.metric(
                                "Benign Claims",
                                total_claims - fraud_detected
                            )
                            
                            avg_fraud_prob = result_df['fraud_probability'].mean()
                            col4.metric(
                                "Avg Fraud Probability",
                                f"{avg_fraud_prob:.2%}"
                            )
                            
                            st.markdown("---")
                            
                            # ========== Results Table ==========
                            st.subheader("🔍 Detection Results")
                            
                            # Filter options
                            filter_col1, filter_col2 = st.columns(2)
                            
                            with filter_col1:
                                show_only = st.selectbox(
                                    "Filter by:",
                                    ["All Claims", "Fraud Only", "Benign Only"]
                                )
                            
                            with filter_col2:
                                if 'predicted_fraud_type' in result_df.columns:
                                    fraud_types = result_df['predicted_fraud_type'].dropna().unique()
                                    selected_type = st.selectbox(
                                        "Fraud Type:",
                                        ["All"] + list(fraud_types)
                                    )
                            
                            # Apply filters
                            filtered_df = result_df.copy()
                            
                            if show_only == "Fraud Only":
                                filtered_df = filtered_df[filtered_df['predicted_fraud'] == 1]
                            elif show_only == "Benign Only":
                                filtered_df = filtered_df[filtered_df['predicted_fraud'] == 0]
                            
                            if 'predicted_fraud_type' in result_df.columns and selected_type != "All":
                                filtered_df = filtered_df[filtered_df['predicted_fraud_type'] == selected_type]
                            
                            # Display key columns
                            display_columns = [
                                'claim_id', 'predicted_fraud', 'fraud_probability', 
                                'predicted_fraud_type', 'billed_amount', 'paid_amount',
                                'claim_ratio', 'diagnosis_name'
                            ]
                            display_columns = [col for col in display_columns if col in filtered_df.columns]
                            
                            st.dataframe(
                                filtered_df[display_columns],
                                use_container_width=True,
                                height=400
                            )
                            
                            st.info(f"Showing {len(filtered_df)} of {len(result_df)} claims")
                            
                            # ========== Download Results ==========
                            st.markdown("---")
                            st.subheader("💾 Download Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download full results
                                csv_full = result_df.to_csv(index=False)
                                st.download_button(
                                    label="⬇️ Download All Results (CSV)",
                                    data=csv_full,
                                    file_name="fraud_detection_results.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Download fraud only
                                fraud_df = result_df[result_df['predicted_fraud'] == 1]
                                csv_fraud = fraud_df.to_csv(index=False)
                                st.download_button(
                                    label="⬇️ Download Fraud Cases Only (CSV)",
                                    data=csv_fraud,
                                    file_name="fraud_cases_only.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    disabled=len(fraud_df) == 0
                                )
                            
                            # ========== RAG AUDIT REPORTS ==========
                            if 'explanation_text' in result_df.columns:
                                st.markdown("---")
                                st.subheader("📋 Detailed Audit Reports (RAG-Generated)")
                                
                                # Get fraud cases with explanations
                                fraud_with_explanation = result_df[
                                    (result_df['predicted_fraud'] == 1) & 
                                    (result_df['explanation_text'].notna()) &
                                    (result_df['explanation_text'] != 'No explanation generated — claim classified as benign.')
                                ].sort_values('fraud_probability', ascending=False)
                                
                                if len(fraud_with_explanation) > 0:
                                    st.success(f"✅ Generated {len(fraud_with_explanation)} detailed audit reports")
                                    
                                    # Display each audit report
                                    for i, (idx, row) in enumerate(fraud_with_explanation.iterrows(), 1):
                                        claim_id = row.get('claim_id', 'Unknown')
                                        fraud_type = row.get('predicted_fraud_type', 'Unknown')
                                        fraud_prob = row.get('fraud_probability', 0)
                                        
                                        with st.expander(
                                            f"📄 Audit Report #{i} - Claim {claim_id} "
                                            f"({fraud_type.upper()}, {fraud_prob:.1%})",
                                            expanded=(i == 1)
                                        ):
                                            explanation_text = row.get('explanation_text', 'No explanation generated')
                                            st.markdown(explanation_text)
                                            
                                            # Financial summary
                                            st.markdown("---")
                                            st.write("**💰 Financial Summary:**")
                                            col1, col2, col3 = st.columns(3)
                                            col1.metric("Billed", f"Rp {row.get('billed_amount', 0):,.0f}")
                                            col2.metric("Paid", f"Rp {row.get('paid_amount', 0):,.0f}")
                                            col3.metric("Claim Ratio", f"{row.get('claim_ratio', 0):.2f}")
                                else:
                                    st.info("ℹ️ No fraud cases detected or explanations not generated")
                        
                        else:
                            st.warning("No predictions found in response")
                    else:
                        st.warning("Unexpected response format")
                        st.json(api_result)
                
                else:
                    st.error(f"❌ Inference failed: {api_result}")
    
    else:
        st.info("👆 Please upload a CSV file to begin fraud detection analysis")

# ==================== MODE 2: SINGLE CLAIM INPUT ====================
else:  # Single Claim Input
    
    st.subheader("✍️ Enter Claim Details")
    
    with st.form("single_claim_form"):
        
        # ========== Basic Information ==========
        st.markdown("#### 📋 Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            claim_id = st.text_input("Claim ID *", value=f"CLAIM-{datetime.now().strftime('%Y%m%d%H%M%S')}")
            nik_hash = st.text_input("Patient NIK Hash *", value="PAT001")
            sex = st.selectbox("Sex *", ["Male", "Female"])
            age = st.number_input("Age *", min_value=0, max_value=120, value=35)
        
        with col2:
            dpjp_id = st.text_input("Provider ID *", value="DOC001")
            faskes_id = st.text_input("Facility ID *", value="FAS001")
            faskes_level = st.selectbox("Facility Level *", ["Tingkat 1", "Tingkat 2", "Tingkat 3"])
            jenis_pelayanan = st.selectbox("Service Type *", ["Rawat Inap", "Rawat Jalan"])
        
        with col3:
            room_class = st.selectbox("Room Class *", ["Kelas 1", "Kelas 2", "Kelas 3", "VIP"])
            lama_dirawat = st.number_input("Length of Stay (days) *", min_value=0, value=3)
            visit_count_30d = st.number_input("Visits (30 days)", min_value=0, value=1)
        
        st.markdown("---")
        
        # ========== Financial Information ==========
        st.markdown("#### 💰 Financial Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            billed_amount = st.number_input("Billed Amount (Rp) *", min_value=0.0, value=5000000.0, step=100000.0)
        
        with col2:
            paid_amount = st.number_input("Paid Amount (Rp) *", min_value=0.0, value=4500000.0, step=100000.0)
        
        with col3:
            drug_cost = st.number_input("Drug Cost (Rp)", min_value=0.0, value=500000.0, step=50000.0)
        
        with col4:
            procedure_cost = st.number_input("Procedure Cost (Rp)", min_value=0.0, value=1000000.0, step=50000.0)
        
        st.markdown("---")
        
        # ========== Clinical Information ==========
        st.markdown("#### 🏥 Clinical Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kode_icd10 = st.text_input("ICD-10 Code", value="J18.9")
            diagnosis_name = st.text_input("Diagnosis Name", value="Pneumonia")
        
        with col2:
            clinical_pathway_name = st.text_input("Clinical Pathway", value="Standard Pneumonia Pathway")
            clinical_pathway_deviation_score = st.slider("Clinical Deviation Score", 0.0, 1.0, 0.2, 0.1)
        
        with col3:
            kapitasi_flag = st.checkbox("Kapitasi Flag")
            referral_flag = st.checkbox("Referral Flag")
            referral_to_same_facility = st.checkbox("Referral to Same Facility")
        
        st.markdown("---")
        
        # ========== Additional Metrics ==========
        st.markdown("#### 📊 Additional Metrics (Optional)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tarif_inacbg = st.number_input("INA-CBG Tariff", min_value=0.0, value=4000000.0, step=100000.0)
        
        with col2:
            provider_monthly_claims = st.number_input("Provider Monthly Claims", min_value=0, value=50)
        
        with col3:
            nik_hash_reuse_count = st.number_input("NIK Reuse Count", min_value=0, value=1)
        
        with col4:
            time_diff_prev_claim = st.number_input("Days Since Previous Claim", min_value=0, value=30)
        
        st.markdown("---")
        
        # ========== Submission ==========
        col1, col2 = st.columns([3, 1])
        
        with col1:
            submitted = st.form_submit_button("🔍 Analyze Single Claim", type="primary", use_container_width=True)
        
        with col2:
            generate_rag_single = st.checkbox("Generate RAG Report", value=True)
    
    # ========== Process Single Claim ==========
    if submitted:
        
        if not is_healthy:
            st.error("❌ Cannot run inference: API is not available")
            st.stop()
        
        # Build claim data dictionary
        claim_data = {
            # Basic info
            'claim_id': claim_id,
            'nik_hash': nik_hash,
            'dpjp_id': dpjp_id,
            'faskes_id': faskes_id,
            'sex': sex,
            'age': age,
            'faskes_level': faskes_level,
            'jenis_pelayanan': jenis_pelayanan,
            'room_class': room_class,
            'lama_dirawat': lama_dirawat,
            'visit_count_30d': visit_count_30d,
            
            # Financial
            'billed_amount': billed_amount,
            'paid_amount': paid_amount,
            'drug_cost': drug_cost,
            'procedure_cost': procedure_cost,
            'tarif_inacbg': tarif_inacbg,
            'selisih_klaim': billed_amount - paid_amount,
            'claim_ratio': billed_amount / paid_amount if paid_amount > 0 else 0,
            'drug_ratio': drug_cost / billed_amount if billed_amount > 0 else 0,
            'procedure_ratio': procedure_cost / billed_amount if billed_amount > 0 else 0,
            
            # Clinical
            'kode_icd10': kode_icd10,
            'diagnosis_name': diagnosis_name,
            'clinical_pathway_name': clinical_pathway_name,
            'clinical_pathway_deviation_score': clinical_pathway_deviation_score,
            'kapitasi_flag': kapitasi_flag,
            'referral_flag': referral_flag,
            'referral_to_same_facility': referral_to_same_facility,
            
            # Additional
            'provider_monthly_claims': provider_monthly_claims,
            'nik_hash_reuse_count': nik_hash_reuse_count,
            'time_diff_prev_claim': time_diff_prev_claim,
            'rolling_avg_cost_30d': billed_amount,  # Placeholder
            'provider_claim_share': 0.1,  # Placeholder
            
            # Location (dummy)
            'provinsi': 'DKI Jakarta',
            'kabupaten': 'Jakarta Pusat'
        }
        
        # Convert to CSV BytesIO
        csv_file = create_csv_from_dict(claim_data)
        csv_file.name = "single_claim.csv"
        
        with st.spinner("🔄 Analyzing claim... This may take a minute..."):
            
            api_success, api_result = run_inference(csv_file, generate_rag_single)
            
            if api_success:
                st.success("✅ Analysis completed successfully!")
                
                # Extract result
                if isinstance(api_result, dict):
                    predictions = api_result.get("predictions", [])
                    
                    if predictions and len(predictions) > 0:
                        result = predictions[0]  # Single claim result
                        
                        # ========== Result Display ==========
                        st.markdown("---")
                        st.subheader("📊 Fraud Detection Result")
                        
                        # Main metrics
                        col1, col2, col3 = st.columns(3)
                        
                        fraud_pred = result.get('predicted_fraud', 0)
                        fraud_prob = result.get('fraud_probability', 0)
                        fraud_type = result.get('predicted_fraud_type', 'Unknown')
                        
                        if fraud_pred == 1:
                            col1.metric("Prediction", "⚠️ FRAUD DETECTED", delta=None)
                        else:
                            col1.metric("Prediction", "✅ BENIGN", delta=None)
                        
                        col2.metric("Fraud Probability", f"{fraud_prob:.1%}")
                        col3.metric("Fraud Type", fraud_type if fraud_pred == 1 else "N/A")
                        
                        # Show claim details
                        st.markdown("---")
                        with st.expander("📋 Claim Details", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Patient & Provider:**")
                                st.write(f"- Patient ID: {nik_hash}")
                                st.write(f"- Provider ID: {dpjp_id}")
                                st.write(f"- Facility: {faskes_id} ({faskes_level})")
                                st.write(f"- Age: {age}, Sex: {sex}")
                                
                                st.write("\n**Clinical:**")
                                st.write(f"- Diagnosis: {diagnosis_name} ({kode_icd10})")
                                st.write(f"- Service Type: {jenis_pelayanan}")
                                st.write(f"- Length of Stay: {lama_dirawat} days")
                                st.write(f"- Deviation Score: {clinical_pathway_deviation_score:.2f}")
                            
                            with col2:
                                st.write("**Financial:**")
                                st.write(f"- Billed: Rp {billed_amount:,.0f}")
                                st.write(f"- Paid: Rp {paid_amount:,.0f}")
                                st.write(f"- Drug Cost: Rp {drug_cost:,.0f}")
                                st.write(f"- Procedure Cost: Rp {procedure_cost:,.0f}")
                                st.write(f"- Claim Ratio: {billed_amount/paid_amount:.2f}" if paid_amount > 0 else "- Claim Ratio: N/A")
                        
                        # Show RAG explanation if fraud
                        if fraud_pred == 1 and 'explanation_text' in result:
                            explanation_text = result.get('explanation_text', '')
                            
                            if explanation_text and explanation_text != 'No explanation generated — claim classified as benign.':
                                st.markdown("---")
                                st.subheader("📋 Detailed Audit Report")
                                
                                st.markdown(explanation_text)
                        
                        # Download single result
                        st.markdown("---")
                        single_result_df = pd.DataFrame([result])
                        csv_result = single_result_df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Result (CSV)",
                            data=csv_result,
                            file_name=f"{claim_id}_result.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    else:
                        st.warning("No prediction result received")
                else:
                    st.warning("Unexpected response format")
                    st.json(api_result)
            
            else:
                st.error(f"❌ Analysis failed: {api_result}")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("BPJS Fraud Detection System v1.1.0 | Powered by FastAPI + GraphXAIN + RAG + Streamlit")
