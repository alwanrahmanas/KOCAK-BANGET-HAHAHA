# ==================== EXPLANATION GENERATOR ====================
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List

from rag_config import RAGConfig
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class ExplanationGenerator:
    """Generate human-readable explanations using LLM + RAG"""

    def __init__(self, config: RAGConfig, retrieval_engine):
        self.config = config
        self.retrieval_engine = retrieval_engine

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )

    # ------------------------------------------------------------------
    def create_explanation_prompt(self, fraud_case: Dict, retrieved_docs: List[Document]):

        explanation_json = fraud_case.get("explanation_json", {})
        if isinstance(explanation_json, str):
            explanation_json = json.loads(explanation_json)

        top_feature_details = explanation_json.get("top_features", [])
        claim = fraud_case.get("claim_data", {})

        regulations_text = "\n\n".join(
            f"**{doc.metadata.get('title','Document')}**:\n{doc.page_content}"
            for doc in retrieved_docs
        )

        # Build the dynamic prompt
        prompt = f"""
Anda adalah auditor BPJS Kesehatan. Jelaskan mengapa klaim berikut terindikasi fraud.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFORMASI KLAIM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Claim ID: {fraud_case.get('claim_id')}
Tipe Fraud: {fraud_case.get('predicted_fraud_type')}
Probabilitas: {fraud_case.get('fraud_probability', 0):.1%}

Diagnosis: {claim.get('diagnosis')}
Prosedur: {claim.get('procedure')}
Jenis Pelayanan: {claim.get('jenis_pelayanan')}
Tarif INA-CBG: {claim.get('tarif_inacbg')}
Tagihan: {claim.get('billed_amount')}
Selisih Klaim: {claim.get('selisih_klaim')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FITUR ANOMALI (SHAP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        for feat in top_feature_details[:5]:
            prompt += f"- {feat.get('feature')} | Value={feat.get('value')} | Impact={feat.get('impact')}\n"

        prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGULASI TERKAIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{regulations_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUKSI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Jelaskan:

1. Ringkasan temuan
2. Analisis fitur kunci yang mencurigakan
3. Pelanggaran regulasi yang relevan
4. Validasi tarif INA-CBG
5. Dampak kerugian potensi
6. Rekomendasi tindak lanjut
"""

        return prompt

    # ------------------------------------------------------------------
    def generate_explanation(self, fraud_case: Dict):
        """Run retrieval + LLM + build report"""

        print(f"\n[Explanation] Claim {fraud_case.get('claim_id')}")

        query = self.retrieval_engine.build_query_from_fraud_case(fraud_case)

        retrieved_docs = self.retrieval_engine.retrieve_with_filters(
            query,
            fraud_type=fraud_case.get("predicted_fraud_type"),
            top_k=self.config.TOP_K_RETRIEVAL
        )

        # Build prompt
        prompt_text = self.create_explanation_prompt(fraud_case, retrieved_docs)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Anda adalah auditor BPJS Kesehatan."),
            ("user", "{input_text}")
        ])

        chain = prompt | self.llm

        response = chain.invoke({"input_text": prompt_text})

        return {
            "claim_id": fraud_case.get("claim_id"),
            "explanation_text": response.content,
            "retrieved_docs": [
                {
                    "title": doc.metadata.get("title"),
                    "type": doc.metadata.get("type"),
                    "snippet": doc.page_content[:200]
                }
                for doc in retrieved_docs
            ],
            "model": self.config.LLM_MODEL,
            "timestamp": datetime.now().isoformat()
        }

    # ------------------------------------------------------------------
    def batch_generate_explanations(self, fraud_cases: List[Dict]):
        results = []

        for item in fraud_cases:
            try:
                results.append(self.generate_explanation(item))
            except Exception as e:
                results.append({
                    "claim_id": item.get("claim_id"),
                    "explanation_text": f"Error: {str(e)}"
                })

        return pd.DataFrame(results)
