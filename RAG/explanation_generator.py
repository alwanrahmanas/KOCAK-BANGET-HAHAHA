# ==================== EXPLANATION GENERATOR ====================
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from RAG.rag_config import RAGConfig
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class ExplanationGenerator:
    """
    🔧 OPTIMIZED: Generate human-readable audit reports using LLM + RAG
    Fully integrated with BPJSFraudInferenceEngine output format
    """

    def __init__(self, config: RAGConfig, retrieval_engine):
        self.config = config
        self.retrieval_engine = retrieval_engine

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )

        print("✅ ExplanationGenerator initialized")
        print(f"   Model: {config.LLM_MODEL}")
        print(f"   Temperature: {config.LLM_TEMPERATURE}")

    def validate_input(self, data) -> pd.DataFrame:
        """
        🔧 NEW: Validate and convert input data to DataFrame
        
        Args:
            data: DataFrame, list of dicts, list of Series, or single dict/Series
        
        Returns:
            Valid DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data
        
        elif isinstance(data, list):
            if not data:
                raise ValueError("Empty list provided")
            
            # Check first element type
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            elif isinstance(data[0], pd.Series):
                return pd.DataFrame(data)
            else:
                raise TypeError(f"List contains unsupported type: {type(data[0])}")
        
        elif isinstance(data, (dict, pd.Series)):
            # Single case - convert to DataFrame with 1 row
            if isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                return pd.DataFrame([data.to_dict()])
        
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

    def print_data_info(self, df: pd.DataFrame):
        """Print diagnostic information about the input data"""
        print(f"\n[Data Info]")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        
        key_columns = [
            'claim_id', 'predicted_fraud', 'predicted_fraud_type',
            'fraud_probability', 'shap_explanation_summary'
        ]
        
        available = [col for col in key_columns if col in df.columns]
        missing = [col for col in key_columns if col not in df.columns]
        
        if available:
            print(f"   ✓ Available key columns: {', '.join(available)}")
        if missing:
            print(f"   ⚠️ Missing columns: {', '.join(missing)}")

    def generate_explanation(self, fraud_case) -> Dict:
        """
        🔧 OPTIMIZED: Generate explanation for a single fraud case
        
        Args:
            fraud_case: pandas Series, dict, or row from inference results
        
        Returns:
            Dict with explanation text and metadata
        """
        # ==================== TYPE HANDLING ====================
        # Convert dict to Series if needed
        if isinstance(fraud_case, dict):
            fraud_case = pd.Series(fraud_case)
        elif not isinstance(fraud_case, pd.Series):
            raise TypeError(f"Expected Series or dict, got {type(fraud_case)}")
        
        claim_id = fraud_case.get('claim_id', 'UNKNOWN')
        
        print(f"\n[Explanation] Processing Claim {claim_id}")

        # Build semantic query from SHAP features
        query = self._build_query_from_shap(fraud_case)

        # Retrieve relevant regulations
        retrieved_docs = self.retrieval_engine.retrieve_with_filters(
            query,
            fraud_type=fraud_case.get("predicted_fraud_type"),
            top_k=self.config.TOP_K_RETRIEVAL
        )

        print(f"   Retrieved {len(retrieved_docs)} regulation documents")

        # Build comprehensive prompt
        prompt_text = self._create_audit_prompt(fraud_case, retrieved_docs)

        # Create LLM chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Anda adalah Auditor Senior BPJS Kesehatan yang berpengalaman dalam investigasi fraud dan compliance regulasi."),
            ("user", "{input_text}")
        ])

        chain = prompt | self.llm

        # Generate explanation
        response = chain.invoke({"input_text": prompt_text})

        return {
            "claim_id": claim_id,
            "explanation_text": response.content,
            "retrieved_docs": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "type": doc.metadata.get("type", "regulation"),
                    "snippet": doc.page_content[:200]
                }
                for doc in retrieved_docs
            ],
            "model": self.config.LLM_MODEL,
            "timestamp": datetime.now().isoformat(),
            "fraud_type": fraud_case.get("predicted_fraud_type"),
            "fraud_probability": fraud_case.get("fraud_probability")
        }

    def batch_generate_explanations(
        self, 
        predictions_data,  # Can be DataFrame, list of dicts, or list of Series
        fraud_only: bool = True
    ) -> pd.DataFrame:
        """
        🔧 OPTIMIZED: Generate explanations for multiple fraud cases
        
        Args:
            predictions_data: DataFrame, list of dicts, or list of Series from inference
            fraud_only: Only process cases where predicted_fraud == 1
        
        Returns:
            DataFrame with explanations
        """
        # ==================== VALIDATE & CONVERT INPUT ====================
        try:
            predictions_df = self.validate_input(predictions_data)
            self.print_data_info(predictions_df)
        except Exception as e:
            print(f"\n❌ Input validation failed: {e}")
            raise
        
        # Filter fraud cases
        if fraud_only:
            if 'predicted_fraud' in predictions_df.columns:
                fraud_mask = predictions_df['predicted_fraud'] == 1
                fraud_cases = predictions_df[fraud_mask]
                print(f"   Filtering: {fraud_mask.sum()} fraud cases out of {len(predictions_df)} total")
            else:
                # If no predicted_fraud column, assume all are fraud cases
                print("   ⚠️ No 'predicted_fraud' column found, processing all cases")
                fraud_cases = predictions_df
        else:
            fraud_cases = predictions_df

        total_cases = len(fraud_cases)
        
        print(f"\n{'='*80}")
        print(f"GENERATING AUDIT REPORTS FOR {total_cases} FRAUD CASES")
        print(f"{'='*80}")

        results = []

        for i, (idx, fraud_case) in enumerate(fraud_cases.iterrows(), 1):
            claim_id = fraud_case.get('claim_id', f'IDX_{idx}')
            
            print(f"\n[{i}/{total_cases}] Processing claim_id: {claim_id}")

            try:
                explanation = self.generate_explanation(fraud_case)
                results.append(explanation)
                
                print(f"   ✓ Explanation generated successfully")

            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
                
                results.append({
                    "claim_id": claim_id,
                    "explanation_text": f"Error generating explanation: {str(e)}",
                    "retrieved_docs": [],
                    "model": self.config.LLM_MODEL,
                    "timestamp": datetime.now().isoformat(),
                    "fraud_type": fraud_case.get("predicted_fraud_type"),
                    "fraud_probability": fraud_case.get("fraud_probability")
                })

        print(f"\n{'='*80}")
        print(f"✅ Completed {len(results)} audit reports")
        print(f"{'='*80}\n")
        
        return pd.DataFrame(results)

    def _build_query_from_shap(self, fraud_case: pd.Series) -> str:
        """
        🔧 NEW: Build semantic query from SHAP explanation summary
        """
        shap_summary = fraud_case.get('shap_explanation_summary', '')
        fraud_type = fraud_case.get('predicted_fraud_type', 'unknown')
        
        # Extract feature names from SHAP summary
        if shap_summary:
            # Format: "feature1↑, feature2↓, feature3↑"
            features = [feat.split('↑')[0].split('↓')[0] for feat in shap_summary.split(',')]
            features_text = ' '.join(features[:5])  # Top 5 features
        else:
            features_text = ''

        # Build comprehensive query
        query_parts = [
            fraud_type,
            features_text,
            fraud_case.get('clinical_pathway_name', ''),
            fraud_case.get('jenis_pelayanan', '')
        ]

        query = ' '.join([str(part) for part in query_parts if part]).strip()
        
        return query if query else fraud_type

    def _create_audit_prompt(
        self, 
        fraud_case: pd.Series, 
        retrieved_docs: List[Document]
    ) -> str:
        """
        🔧 IMPROVED: Create comprehensive audit prompt with better instructions
        """
        
        # ==================== SAFE GETTER HELPERS ====================
        def safe_get(key: str, default='N/A'):
            """Safely get value with fallback"""
            val = fraud_case.get(key, default)
            return val if val is not None and str(val).strip() != '' else default
        
        def safe_float(key: str, default=0.0):
            """Safely get numeric value"""
            try:
                val = fraud_case.get(key, default)
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        def safe_int(key: str, default=0):
            """Safely get integer value"""
            try:
                val = fraud_case.get(key, default)
                return int(float(val)) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        # ==================== EXTRACT CORE DATA ====================
        claim_id = safe_get("claim_id", "UNKNOWN")
        predicted_fraud_type = safe_get("predicted_fraud_type", "unknown")
        fraud_probability = safe_float("fraud_probability", 0)
        fraud_label = safe_get("fraud_label", "UNKNOWN")
        
        # ==================== CLINICAL DATA ====================
        kode_icd10 = safe_get("kode_icd10")
        clinical_pathway_name = safe_get("clinical_pathway_name")
        diagnosis_name = safe_get("diagnosis_name")
        kode_prosedur = safe_get("kode_prosedur")
        procedure_name = safe_get("procedure_name")
        
        # ==================== FINANCIAL DATA ====================
        inacbg_code = safe_get("inacbg_code")
        kode_tarif_inacbg = safe_get("kode_tarif_inacbg")
        tarif_inacbg = safe_float("tarif_inacbg")
        billed_amount = safe_float("billed_amount")
        paid_amount = safe_float("paid_amount")
        selisih_klaim = safe_float("selisih_klaim")
        claim_ratio = safe_float("claim_ratio")
        
        # ==================== SERVICE DATA ====================
        jenis_pelayanan = safe_get("jenis_pelayanan")
        room_class = safe_get("room_class")
        lama_dirawat = safe_int("lama_dirawat")
        
        # ==================== RATIO INDICATORS ====================
        drug_ratio = safe_float("drug_ratio")
        procedure_ratio = safe_float("procedure_ratio")
        
        # ==================== PROVIDER DATA ====================
        faskes_id = safe_get("faskes_id")
        faskes_level = safe_get("faskes_level")
        provider_monthly_claims = safe_int("provider_monthly_claims")
        
        # ==================== PATIENT HISTORY ====================
        visit_count_30d = safe_int("visit_count_30d")
        clinical_pathway_score = safe_float("clinical_pathway_deviation_score")
        
        # ==================== SHAP EXPLANATION ====================
        shap_summary = safe_get("shap_explanation_summary", "Tidak tersedia")
        shap_top_features = safe_get("shap_top_features", [])
        
        # Parse SHAP features if string
        if isinstance(shap_top_features, str):
            try:
                shap_top_features = json.loads(shap_top_features.replace("'", '"'))
            except:
                shap_top_features = []
        
        # Format SHAP features
        shap_details = self._format_shap_features_improved(shap_top_features, predicted_fraud_type)
        
        # ==================== REGULATION DOCUMENTS ====================
        regulations_text = self._format_regulations(retrieved_docs)
        
        # ==================== BUILD IMPROVED PROMPT ====================
        prompt = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    LAPORAN AUDIT KLAIM BPJS KESEHATAN
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    <CLAIM_DATA>
    # Identitas Klaim
    claim_id: {claim_id}
    fraud_type: {predicted_fraud_type}
    fraud_probability: {fraud_probability:.4f}
    fraud_label: {fraud_label}

    # Data Klinis
    icd10: {kode_icd10}
    diagnosis_name: {diagnosis_name}
    clinical_pathway: {clinical_pathway_name}
    procedure_code: {kode_prosedur}
    procedure_name: {procedure_name}
    clinical_pathway_deviation_score: {clinical_pathway_score}

    # Data Finansial
    inacbg_code: {inacbg_code}
    kode_tarif_inacbg: {kode_tarif_inacbg}
    tarif_inacbg: {tarif_inacbg:,.0f}
    billed_amount: {billed_amount:,.0f}
    paid_amount: {paid_amount:,.0f}
    selisih_klaim: {selisih_klaim:,.0f}
    claim_ratio: {claim_ratio:.4f}

    # Komposisi Biaya
    drug_ratio: {drug_ratio:.4f}
    procedure_ratio: {procedure_ratio:.4f}

    # Data Pelayanan
    jenis_pelayanan: {jenis_pelayanan}
    room_class: {room_class}
    lama_dirawat: {lama_dirawat} hari

    # Data Provider & Pasien
    faskes_id: {faskes_id}
    faskes_level: {faskes_level}
    provider_monthly_claims: {provider_monthly_claims}
    visit_count_30d: {visit_count_30d}

    # SHAP Explanation
    shap_summary: {shap_summary}
    </CLAIM_DATA>

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SHAP FEATURE ANALYSIS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {shap_details}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    REGULASI & REFERENSI
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {regulations_text}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    INSTRUKSI AUDIT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    PRINSIP UTAMA:
    1. Gunakan HANYA data di dalam <CLAIM_DATA>
    2. DILARANG membuat/mengada-ada data yang tidak tersedia
    3. Jika data kosong/N/A → tulis "Tidak ada data" dan skip analisis terkait
    4. Fokus pada fitur SHAP yang tersedia dan relevan dengan {predicted_fraud_type}
    5. Setiap angka HARUS dari <CLAIM_DATA> (jangan rounded atau estimasi)

    FORMAT LAPORAN:

    1. EXECUTIVE SUMMARY (maksimal 4 poin bullet)
    • Ringkasan temuan utama dengan claim_id dan fraud type
    • Status risiko fraud (TINGGI/SEDANG/RENDAH) berdasarkan fraud_probability
    • Estimasi kerugian potensial (jika ada selisih_klaim)
    • Rekomendasi utama (1 kalimat)

    2. ANALISIS FINANSIAL
    Bandingkan: tarif_inacbg vs billed_amount vs paid_amount
    • Apakah ada overclaim/underclaim? (berdasarkan selisih_klaim)
    • Evaluasi claim_ratio (normal: 0.8-1.2, suspicious: >1.5 atau <0.5)
    • Analisis komposisi biaya (drug_ratio, procedure_ratio)
    • Hitung persentase deviasi dari tarif INA-CBG
    **WAJIB: Sebutkan angka ASLI dari data, jangan dibulatkan**

    3. ANALISIS KLINIS
    • Diagnosis (ICD-10: {kode_icd10}) dan kesesuaiannya dengan prosedur
    • Evaluasi lama_dirawat vs clinical_pathway_deviation_score
    • Apakah prosedur yang tercatat sesuai dengan diagnosis?
    • Cross-check dengan guideline klinis (jika ada di regulasi)
    **Jika diagnosis/prosedur = N/A, skip detail dan tulis "Data tidak tersedia"**

    4. VALIDASI INA-CBG
    • Verifikasi kode INA-CBG: {inacbg_code} dengan diagnosis/prosedur
    • Bandingkan tarif_inacbg dengan billed_amount
    • Apakah ada indikasi upcoding/downcoding?
    • Rujuk regulasi INA-CBG (jika tersedia)

    5. ANALISIS FRAUD TYPE: {predicted_fraud_type}
    Definisi fraud type ini menurut regulasi:
    {self._get_fraud_type_definition(predicted_fraud_type)}
    
    Analisis spesifik untuk {predicted_fraud_type}:
    • Bukti numerik yang mendukung (dari SHAP features)
    • Pola yang teridentifikasi (dari data klinis/finansial)
    • Regulasi yang berpotensi dilanggar (kutip dari REGULASI & REFERENSI)
    • Tingkat keyakinan berdasarkan fraud_probability ({fraud_probability:.1%})

    6. INTERPRETASI MACHINE LEARNING
    Analisis fitur SHAP teratas:
    

    7. POLA PROVIDER & PASIEN
    • Volume klaim provider: {provider_monthly_claims} klaim/bulan (normal: <300)
    • Frekuensi kunjungan pasien: {visit_count_30d} kali/30 hari (normal: <3)
    • Tingkat faskes: {faskes_level}
    • Red flags historis (jika visit_count_30d tinggi)

    8. TEMUAN BERDASARKAN REGULASI
    • Kutip pasal/ayat spesifik dari REGULASI & REFERENSI yang relevan
    • Identifikasi pelanggaran konkret (jika ada)
    • Tingkat keparahan: RENDAH/SEDANG/TINGGI/KRITIS

    9. KESIMPULAN AUDIT
    • Ringkasan 3-5 kalimat
    • Tingkat keyakinan fraud: Model ({fraud_probability:.1%}) vs Manual Audit
    • Estimasi kerugian negara (jika fraud terbukti)
    • Risk score final: RENDAH/SEDANG/TINGGI/KRITIS

    10. REKOMENDASI TINDAK LANJUT
        A. Investigasi Lanjutan:
        • Dokumen spesifik yang perlu diminta (rekam medis, kuitansi, dll)
        • Data tambahan yang diperlukan
        
        B. Sanksi yang Direkomendasikan:
        • Sesuai PerBPJS No. 6/2020 (jika ada pelanggaran)
        • Estimasi nilai yang harus dikembalikan
        
        C. Langkah Pencegahan:
        • Training untuk coder/billing staff
        • Kontrol sistem yang perlu diperkuat
        • Monitoring berkelanjutan

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CRITICAL REMINDERS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ❌ JANGAN membuat data yang tidak ada di <CLAIM_DATA>
    ❌ JANGAN membulatkan angka finansial (gunakan angka asli)
    ❌ JANGAN mengabaikan nilai N/A (explicitly state "Tidak ada data")
    ✓ Gunakan fraud_probability sebagai baseline confidence
    ✓ Prioritaskan fitur SHAP dengan nilai absolut terbesar
    ✓ Setiap klaim regulasi HARUS ada kutipan dari REGULASI & REFERENSI
    ✓ Jika data tidak cukup untuk kesimpulan → state "Perlu investigasi lanjutan"

    Mulai analisis:
    """
        
        return prompt


    def _format_shap_features_improved(self, shap_features, fraud_type: str) -> str:
        """
        🔧 NEW: Format SHAP features with fraud-type specific interpretation
        """
        if not shap_features:
            return "⚠️ Data SHAP tidak tersedia - analisis ML tidak dapat dilakukan."
        
        # Handle string input
        if isinstance(shap_features, str):
            try:
                shap_features = json.loads(shap_features.replace("'", '"'))
            except:
                return f"⚠️ Data SHAP tidak dapat diparse: {shap_features[:100]}..."
        
        # Handle list input
        if not isinstance(shap_features, list):
            return f"⚠️ Format SHAP tidak didukung: {type(shap_features)}"
        
        formatted = []
        formatted.append(f"Fraud Type Context: {fraud_type}\n")
        formatted.append("Analisis fitur ML yang berkontribusi terhadap prediksi fraud:\n")
        
        for i, feat in enumerate(shap_features[:5], 1):
            if not isinstance(feat, dict):
                continue
                
            feature_name = feat.get('feature', 'unknown')
            feature_value = feat.get('value', 'N/A')
            shap_value = feat.get('shap_value', 0)
            direction = feat.get('direction', '?')
            
            # Interpret direction
            impact = "MENINGKATKAN" if direction == '↑' else "MENURUNKAN" if direction == '↓' else "NETRAL"
            
            # Feature-specific interpretation
            interpretation = self._interpret_feature_for_fraud_type(
                feature_name, feature_value, shap_value, fraud_type
            )
            
            try:
                formatted.append(
                    f"{i}. {feature_name} {direction} | SHAP: {float(shap_value):.4f}\n"
                    f"   • Nilai: {feature_value}\n"
                    f"   • Impact: {impact} risiko fraud\n"
                    f"   • Interpretasi: {interpretation}\n"
                )
            except (ValueError, TypeError):
                formatted.append(
                    f"{i}. {feature_name} {direction}\n"
                    f"   • Nilai: {feature_value}\n"
                    f"   • SHAP: {shap_value}\n"
                    f"   • Interpretasi: {interpretation}\n"
                )
        
        return '\n'.join(formatted)


    def _interpret_feature_for_fraud_type(
        self, 
        feature: str, 
        value, 
        shap_value: float,
        fraud_type: str
    ) -> str:
        """
        🔧 NEW: Provide fraud-type specific interpretation for each feature
        """
        interpretations = {
            'upcoding_diagnosis': {
                'claim_ratio': 'Rasio klaim tinggi dapat indikasi diagnosis dikode lebih tinggi untuk tarif lebih besar',
                'billed_amount': 'Tagihan tinggi tidak proporsional dengan layanan dapat indikasi upcoding',
                'clinical_pathway_deviation_score': 'Deviasi pathway tinggi dapat indikasi diagnosis tidak sesuai',
                'tarif_inacbg': 'Gap antara tarif dan tagihan perlu evaluasi kesesuaian kode diagnosis',
            },
            'inflated_bill': {
                'claim_ratio': 'Rasio >1.5 strong indicator inflated billing',
                'billed_amount': 'Tagihan jauh melebihi tarif standar adalah red flag utama',
                'selisih_klaim': 'Selisih besar indikasi markup berlebihan',
                'drug_ratio': 'Proporsi obat tinggi dapat indikasi markup obat',
                'procedure_ratio': 'Proporsi tindakan tinggi dapat indikasi markup prosedur',
            },
            'phantom_billing': {
                'visit_count_30d': 'Kunjungan berulang tinggi tanpa justifikasi medis perlu investigasi',
                'provider_monthly_claims': 'Volume klaim provider sangat tinggi dapat indikasi phantom claims',
                'clinical_pathway_deviation_score': 'Deviasi tinggi dapat indikasi layanan tidak sesuai/fiktif',
            },
        }
        
        fraud_specific = interpretations.get(fraud_type, {})
        return fraud_specific.get(feature, 'Perlu analisis lebih lanjut terhadap fitur ini')


    def _get_fraud_type_definition(self, fraud_type: str) -> str:
        """
        🔧 NEW: Get regulatory definition for each fraud type
        """
        definitions = {
            'upcoding_diagnosis': """
    Upcoding diagnosis adalah praktik mengkode diagnosis dengan tingkat keparahan 
    atau kelompok tarif yang lebih tinggi dari kondisi aktual pasien untuk 
    meningkatkan pembayaran klaim (PerBPJS No. 6/2020 Pasal 15).
    Indikator: claim_ratio >1.2, clinical_pathway_deviation_score tinggi, 
    gap antara layanan aktual vs diagnosis tercatat.
    """,
            'inflated_bill': """
    Inflated billing adalah praktik menaikkan komponen biaya (obat, tindakan, 
    atau layanan penunjang) melebihi harga wajar atau tarif standar yang berlaku.
    Indikator: claim_ratio >1.5, drug_ratio atau procedure_ratio tidak wajar,
    selisih_klaim besar tanpa justifikasi.
    """,
            'phantom_billing': """
    Phantom billing adalah penagihan untuk layanan yang tidak pernah diberikan
    atau pasien fiktif (PerBPJS No. 6/2020 Pasal 14).
    Indikator: visit_count_30d sangat tinggi, provider_monthly_claims abnormal,
    data klinis tidak konsisten dengan klaim.
    """,
    'cloning_claim': """
        Cloning claim adalah praktik penjiplakan klaim dari pasien lain, 
        yaitu klaim yang dibuat dengan cara menyalin (copy-paste) seluruh atau sebagian data 
        atau rekam medis dari pasien/klaim lain yang sudah ada (Peraturan BPJS Kesehatan No 7 Tahun 2016).

        Indikator cloning claim biasanya berupa kemiripan pola data klinis, 
        prosedur, dan rincian tagihan antara dua atau lebih klaim berbeda, 
        serta terdapat duplikasi identitas atau episode pelayanan. 
        Modus ini mengakibatkan klaim ganda atas layanan yang hanya dilakukan satu kali,
        dan sering sulit terdeteksi dalam audit manual biasa.

        Sanksi terhadap cloning claim diatur dalam PerBPJS No. 7/2016 
        dan Undang-Undang Kesehatan No. 17/2023 sebagai bentuk kecurangan serius dalam pengelolaan 
        dana Jaminan Kesehatan Nasional.
        """,

        }
        
        return definitions.get(fraud_type, "Definisi fraud type tidak tersedia dalam database.")


    def _create_shap_interpretation_guide(self, shap_features, fraud_type: str) -> str:
        """
        🔧 NEW: Create interpretation guide for SHAP features
        """
        if not shap_features or not isinstance(shap_features, list):
            return "Data SHAP tidak tersedia untuk interpretasi."
        
        guide = []
        guide.append(f"Panduan interpretasi untuk {fraud_type}:\n")
        
        for feat in shap_features[:5]:
            if isinstance(feat, dict):
                feature = feat.get('feature', 'unknown')
                shap_val = feat.get('shap_value', 0)
                direction = feat.get('direction', '?')
                
                if abs(float(shap_val)) > 1.0:
                    strength = "SANGAT KUAT"
                elif abs(float(shap_val)) > 0.5:
                    strength = "KUAT"
                else:
                    strength = "MODERATE"
                
                guide.append(
                    f"• {feature}: Kontribusi {strength} ({shap_val:.3f}) → "
                    f"Perhatikan apakah nilai aktual konsisten dengan {fraud_type}"
                )
        
        return '\n'.join(guide)

    def _format_shap_features(self, shap_features) -> str:
        """Format SHAP features for display with robust error handling"""
        if not shap_features:
            return "Tidak ada data SHAP yang tersedia."
        
        # Handle string input
        if isinstance(shap_features, str):
            try:
                shap_features = json.loads(shap_features.replace("'", '"'))
            except:
                return f"Data SHAP tidak dapat diparse: {shap_features[:100]}..."
        
        # Handle list input
        if not isinstance(shap_features, list):
            return f"Format SHAP tidak didukung: {type(shap_features)}"
        
        formatted = []
        for i, feat in enumerate(shap_features[:5], 1):
            if not isinstance(feat, dict):
                continue
                
            feature_name = feat.get('feature', 'unknown')
            feature_value = feat.get('value', 0)
            
            # Handle different SHAP value keys
            shap_value = feat.get('shap_value') or feat.get('impact', 0)
            
            direction = feat.get('direction', '?')
            
            try:
                formatted.append(
                    f"{i}. {feature_name} {direction}\n"
                    f"   Nilai: {float(feature_value):.2f}\n"
                    f"   SHAP Impact: {float(shap_value):.4f}\n"
                )
            except (ValueError, TypeError) as e:
                formatted.append(
                    f"{i}. {feature_name} {direction}\n"
                    f"   Nilai: {feature_value}\n"
                    f"   SHAP Impact: {shap_value}\n"
                )
        
        return '\n'.join(formatted) if formatted else "Tidak ada fitur SHAP yang valid."

    def _format_regulations(self, docs: List[Document]) -> str:
        """Format regulation documents for display"""
        if not docs:
            return "Tidak ada regulasi relevan yang ditemukan melalui RAG retrieval."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', f'Dokumen {i}')
            content = doc.page_content[:500]  # Limit content length
            
            formatted.append(
                f"[{i}] {title}\n"
                f"{content}...\n"
            )
        
        return '\n'.join(formatted)

    def _calculate_anomaly_indicators(self, fraud_case: pd.Series) -> str:
        """Calculate and format anomaly indicators with safe numeric handling"""
        indicators = []
        
        def safe_float(value, default=0.0):
            """Safely convert to float"""
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Claim ratio anomaly
        claim_ratio = safe_float(fraud_case.get('claim_ratio', 0))
        if claim_ratio > 1.2:
            indicators.append(f"⚠️ Claim Ratio TINGGI: {claim_ratio:.2f} (>120% dari tarif)")
        elif claim_ratio > 1.0:
            indicators.append(f"⚠️ Claim Ratio: {claim_ratio:.2f} (>100% dari tarif)")
        elif claim_ratio < 0.8 and claim_ratio > 0:
            indicators.append(f"⚠️ Claim Ratio RENDAH: {claim_ratio:.2f} (<80% dari tarif)")
        elif claim_ratio > 0:
            indicators.append(f"✓ Claim Ratio normal: {claim_ratio:.2f}")
        
        # Visit frequency anomaly
        visit_count = safe_float(fraud_case.get('visit_count_30d', 0))
        if visit_count > 5:
            indicators.append(f"⚠️ Kunjungan Berulang: {int(visit_count)} kali dalam 30 hari")
        
        # LOS anomaly
        lama_dirawat = safe_float(fraud_case.get('lama_dirawat', 0))
        if lama_dirawat > 14:
            indicators.append(f"⚠️ Lama Rawat Panjang: {int(lama_dirawat)} hari")
        elif lama_dirawat > 7:
            indicators.append(f"⚠️ Lama Rawat: {int(lama_dirawat)} hari (perlu review)")
        
        # Drug ratio anomaly
        drug_ratio = safe_float(fraud_case.get('drug_ratio', 0))
        if drug_ratio > 0.5:
            indicators.append(f"⚠️ Drug Ratio TINGGI: {drug_ratio:.2%}")
        
        # Provider volume anomaly
        provider_claims = safe_float(fraud_case.get('provider_monthly_claims', 0))
        if provider_claims > 500:
            indicators.append(f"⚠️ Volume Klaim Provider Tinggi: {int(provider_claims)} klaim/bulan")
        
        if not indicators:
            indicators.append("✓ Tidak ada anomali numerik yang signifikan terdeteksi")
        
        return '\n'.join([f"• {ind}" for ind in indicators])

    def export_explanations(
        self, 
        explanations_df: pd.DataFrame, 
        output_path: str,
        format: str = 'excel'
    ):
        """
        Export explanations to file
        
        Args:
            explanations_df: DataFrame with explanations
            output_path: Output file path
            format: 'excel', 'csv', or 'json'
        """
        print(f"\n[Export] Saving {len(explanations_df)} explanations to {output_path}")
        
        try:
            if format == 'excel':
                explanations_df.to_excel(output_path, index=False, engine='openpyxl')
            elif format == 'csv':
                explanations_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif format == 'json':
                explanations_df.to_json(output_path, orient='records', indent=2, force_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"   ✓ Export completed: {output_path}")
        except Exception as e:
            print(f"   ✗ Export failed: {e}")
            raise

    def get_summary_stats(self, explanations_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics from explanations
        
        Returns:
            Dictionary with summary metrics
        """
        if explanations_df.empty:
            return {"total_cases": 0, "status": "empty"}
        
        stats = {
            "total_cases": len(explanations_df),
            "unique_fraud_types": explanations_df['fraud_type'].nunique() if 'fraud_type' in explanations_df.columns else 0,
            "avg_fraud_probability": explanations_df['fraud_probability'].mean() if 'fraud_probability' in explanations_df.columns else 0,
            "timestamp_range": {
                "first": explanations_df['timestamp'].min() if 'timestamp' in explanations_df.columns else None,
                "last": explanations_df['timestamp'].max() if 'timestamp' in explanations_df.columns else None
            }
        }
        
        if 'fraud_type' in explanations_df.columns:
            stats['fraud_type_distribution'] = explanations_df['fraud_type'].value_counts().to_dict()
        
        return stats