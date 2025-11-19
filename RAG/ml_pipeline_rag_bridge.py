# ==================== INTEGRATION BRIDGE ====================
import json
import pandas as pd
from typing import List, Dict


class MLPipelineRAGBridge:
    """
    Bridge between ML Pipeline and RAG System.
    Converts ML prediction outputs → RAG-ready fraud case dicts.
    """

    @staticmethod
    def prepare_fraud_cases_for_rag(
        predictions_df: pd.DataFrame,
        original_claims_df: pd.DataFrame,
        fraud_only: bool = True
    ) -> List[Dict]:

        merged = predictions_df.merge(
            original_claims_df,
            on="claim_id",
            how="left",
            suffixes=('', '_orig')  # Handle duplicate columns
        )

        if fraud_only:
            merged = merged[merged["predicted_fraud"] == 1].copy()

        fraud_cases = []

        for _, row in merged.iterrows():

            # Parse JSON safely
            explanation_json = row.get("explanation_json", {})
            if isinstance(explanation_json, str):
                try:
                    explanation_json = json.loads(explanation_json)
                except Exception:
                    explanation_json = {}

            # ========== ENHANCED CLAIM DATA (ALL IMPORTANT COLUMNS) ==========
            fraud_case = {
                "claim_id": row["claim_id"],
                "predicted_fraud": int(row.get("predicted_fraud", 0)),
                "fraud_probability": float(row.get("fraud_probability", 0.0)),
                "predicted_fraud_type": row.get("predicted_fraud_type", "N/A"),
                "top_features": row.get("top_features", ""),
                "explanation_summary": row.get("explanation_summary", ""),
                "explanation_json": explanation_json,

                "claim_data": {
                    # ========== DIAGNOSIS & PROCEDURE (CRITICAL) ==========
                    "diagnosis_code": row.get("kode_icd10", row.get("diagnosis_code", None)),
                    "diagnosis_name": row.get("diagnosis_name", None),
                    "procedure_code": row.get("kode_prosedur", row.get("procedure_code", None)),
                    "procedure_name": row.get("procedure_name", None),
                    
                    # ========== INA-CBG TARIFF (CRITICAL FOR VALIDATION) ==========
                    "inacbg_code": row.get("kode_tarif_inacbg", row.get("inacbg_code", None)),
                    "tarif_inacbg": row.get("tarif_inacbg", 0),
                    "inacbg_group": row.get("inacbg_group", None),
                    "severity_level": row.get("severity_level", None),
                    
                    # ========== SERVICE TYPE & CLASS ==========
                    "jenis_pelayanan": row.get("jenis_pelayanan", None),
                    "room_class": row.get("room_class", None),
                    "service_type": row.get("service_type", None),
                    
                    # ========== FINANCIAL DATA ==========
                    "billed_amount": row.get("billed_amount", 0),
                    "paid_amount": row.get("paid_amount", 0),
                    "selisih_klaim": row.get("selisih_klaim", 0),
                    "claim_ratio": row.get("claim_ratio", 0),
                    
                    # ========== COST BREAKDOWN ==========
                    "drug_cost": row.get("drug_cost", 0),
                    "procedure_cost": row.get("procedure_cost", 0),
                    "device_cost": row.get("device_cost", 0),
                    "consumable_cost": row.get("consumable_cost", 0),
                    "service_cost": row.get("service_cost", 0),
                    "drug_ratio": row.get("drug_ratio", 0),
                    "procedure_ratio": row.get("procedure_ratio", 0),
                    
                    # ========== CLINICAL DATA ==========
                    "lama_dirawat": row.get("lama_dirawat", None),
                    "los_deviation": row.get("los_deviation", 0),
                    "clinical_pathway_name": row.get("clinical_pathway_name", None),
                    "clinical_pathway_deviation_score": row.get("clinical_pathway_deviation_score", 0),
                    
                    # ========== PROVIDER INFO ==========
                    "faskes_id": row.get("faskes_id", None),
                    "faskes_name": row.get("faskes_name", None),
                    "faskes_level": row.get("faskes_level", None),
                    "dpjp_id": row.get("dpjp_id", None),
                    "dpjp_name": row.get("dpjp_name", None),
                    "provinsi": row.get("provinsi", None),
                    "kabupaten": row.get("kabupaten", None),
                    
                    # ========== PATIENT INFO ==========
                    "age": row.get("age", None),
                    "sex": row.get("sex", None),
                    "participant_type": row.get("participant_type", None),
                    
                    # ========== TEMPORAL PATTERNS ==========
                    "admission_date": row.get("admission_date", None),
                    "discharge_date": row.get("discharge_date", None),
                    "claim_date": row.get("claim_date", None),
                    "visit_count_30d": row.get("visit_count_30d", 0),
                    "visit_count_90d": row.get("visit_count_90d", 0),
                    
                    # ========== PROVIDER PATTERNS ==========
                    "provider_monthly_claims": row.get("provider_monthly_claims", 0),
                    "provider_avg_claim_amount": row.get("provider_avg_claim_amount", 0),
                    "provider_claim_share": row.get("provider_claim_share", 0),
                    
                    # ========== ADDITIONAL CONTEXT ==========
                    "emergency_case": row.get("emergency_case", None),
                    "referral_type": row.get("referral_type", None),
                    "discharge_status": row.get("discharge_status", None),
                },
            }
            # ==================================================================

            # Optional multi-label fraud predictions
            if "predicted_fraud_types_multi" in row:
                fraud_case["predicted_fraud_types_multi"] = row["predicted_fraud_types_multi"]

            fraud_cases.append(fraud_case)

        return fraud_cases

    # ----------------------------------------------------------------------

    @staticmethod
    def merge_rag_explanations_back(
        predictions_df: pd.DataFrame,
        rag_explanations_df: pd.DataFrame
    ) -> pd.DataFrame:

        merged = predictions_df.merge(
            rag_explanations_df[
                ["claim_id", "explanation_text", "retrieved_docs", "model", "timestamp"]
            ],
            on="claim_id",
            how="left"
        )

        merged["explanation_text"] = merged["explanation_text"].fillna(
            "No explanation generated — claim classified as benign."
        )

        return merged
