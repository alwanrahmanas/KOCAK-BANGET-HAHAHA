# ==================== RETRIEVAL ENGINE ====================
from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from rag_config import RAGConfig
from collections import Counter
import re


class RetrievalEngine:
    """
    Smart hybrid retrieval with:
    - Semantic similarity search
    - Metadata filtering (key_topics, entities, contains_rule)
    - Multi-stage re-ranking
    - Query expansion
    """

    def __init__(self, vector_store: SupabaseVectorStore, config: RAGConfig):
        self.vector_store = vector_store
        self.config = config

    # ========== QUERY BUILDING ==========
    
    def build_query_from_fraud_case(self, fraud_case: Dict) -> str:
        """
        Build optimized retrieval query from fraud case
        Enhanced with medical terminology and regulatory keywords
        """
        query_parts = []

        # 1. Fraud type (primary signal)
        fraud_type = fraud_case.get("predicted_fraud_type", "")
        if fraud_type and fraud_type != "benign":
            # Map fraud types to regulatory terms
            fraud_terms = {
                "inflated_bill": "tarif klaim berlebihan, markup biaya, inflasi tagihan",
                "upcoding_diagnosis": "upcoding diagnosis, peningkatan kode ICD, manipulasi diagnosis",
                "service_unbundling": "unbundling layanan, pemisahan tindakan, fragmentasi prosedur",
                "phantom_billing": "tagihan fiktif, layanan tidak diberikan, klaim palsu",
                "prolonged_los": "perpanjangan rawat inap, LOS tidak wajar, length of stay berlebihan",
                "room_manipulation": "manipulasi kelas perawatan, upgrade ruangan tidak sesuai",
                "unnecessary_services": "layanan tidak perlu, overutilization, tindakan berlebihan",
                "repeat_billing": "tagihan ganda, duplikasi klaim, double billing",
            }
            fraud_keywords = fraud_terms.get(fraud_type, fraud_type.replace("_", " "))
            query_parts.append(f"{fraud_type}: {fraud_keywords}")

        # 2. Top features from SHAP (high impact)
        top_features = fraud_case.get("top_features", "")
        if top_features:
            # Parse features and map to domain concepts
            feature_mapping = {
                "claim_ratio": "rasio klaim terhadap tarif INA-CBG, perbandingan tagihan",
                "selisih_klaim": "selisih klaim, perbedaan tarif, markup",
                "lama_dirawat": "lama rawat inap, length of stay, LOS",
                "drug_cost": "biaya obat, pharmaceutical cost, drug pricing",
                "procedure_cost": "biaya tindakan, procedure fees",
                "drug_ratio": "rasio biaya obat, drug cost ratio",
                "procedure_ratio": "rasio biaya prosedur",
            }
            
            for feat, keywords in feature_mapping.items():
                if feat in top_features:
                    query_parts.append(keywords)

        # 3. Clinical context
        claim_data = fraud_case.get("claim_data", {})
        
        if claim_data.get("diagnosis"):
            diagnosis = claim_data["diagnosis"]
            # Extract ICD code if present
            icd_match = re.search(r'[A-Z]\d{2}(\.\d)?', diagnosis)
            if icd_match:
                query_parts.append(f"ICD {icd_match.group(0)}")
            query_parts.append(f"diagnosis {diagnosis}")

        if claim_data.get("procedure"):
            procedure = claim_data["procedure"]
            query_parts.append(f"prosedur {procedure}")

        if claim_data.get("jenis_pelayanan"):
            service_type = claim_data["jenis_pelayanan"]
            query_parts.append(f"jenis pelayanan {service_type}")

        # 4. Fraud-specific regulatory keywords
        if "claim_ratio" in top_features or "selisih_klaim" in top_features:
            query_parts.append("validasi tarif INA-CBG, batas klaim maksimal, standar billing")

        if "lama_dirawat" in top_features:
            query_parts.append("standar lama rawat, typical LOS, clinical pathway")

        if "drug_cost" in top_features or "drug_ratio" in top_features:
            query_parts.append("regulasi obat, formularium nasional, batas biaya farmasi")
        
        if "procedure_cost" in top_features or "procedure_ratio" in top_features:
            query_parts.append("tarif tindakan medis, standar prosedur, biaya intervensi")

        # 5. Add general fraud detection context
        query_parts.append("deteksi fraud, verifikasi klaim, audit BPJS")

        query = " | ".join(query_parts)
        return query

    # ========== METADATA-BASED FILTERING ==========
    
    def _extract_metadata_filters(self, fraud_case: Dict) -> Dict[str, List[str]]:
        """
        Extract metadata filters from fraud case
        Returns dict with filter criteria for key_topics, entities, etc.
        """
        filters = {
            "required_topics": [],
            "preferred_entities": [],
            "must_contain_rules": False
        }

        fraud_type = fraud_case.get("predicted_fraud_type", "")
        
        # Map fraud types to required topics
        fraud_topic_map = {
            "inflated_bill": ["tariff_validation", "billing_standards", "inacbg_rates"],
            "upcoding_diagnosis": ["diagnosis_groups", "coding_guidelines", "fraud_penalties"],
            "service_unbundling": ["inacbg_codes", "bundling_rules", "procedure_standards"],
            "prolonged_los": ["clinical_pathways", "length_of_stay", "admission_criteria"],
            "unnecessary_services": ["medical_necessity", "treatment_standards", "overutilization"],
        }
        
        if fraud_type in fraud_topic_map:
            filters["required_topics"] = fraud_topic_map[fraud_type]

        # Extract ICD codes, procedures from claim data
        claim_data = fraud_case.get("claim_data", {})
        
        if claim_data.get("diagnosis"):
            icd_match = re.search(r'[A-Z]\d{2}(\.\d)?', claim_data["diagnosis"])
            if icd_match:
                filters["preferred_entities"].append(icd_match.group(0))
        
        # For billing fraud, we MUST have rules/thresholds
        if fraud_type in ["inflated_bill", "upcoding_diagnosis", "service_unbundling"]:
            filters["must_contain_rules"] = True

        return filters

    def _apply_metadata_filters(
        self, 
        documents: List[Tuple[Document, float]], 
        filters: Dict[str, List[str]]
    ) -> List[Tuple[Document, float]]:
        """
        Filter and boost documents based on metadata
        """
        filtered = []
        
        for doc, score in documents:
            boost_factor = 1.0
            
            # Check if doc contains required topics
            doc_topics = doc.metadata.get("key_topics", [])
            required_topics = filters.get("required_topics", [])
            
            if required_topics:
                topic_overlap = len(set(doc_topics) & set(required_topics))
                if topic_overlap > 0:
                    boost_factor *= (1.0 + 0.2 * topic_overlap)  # +20% per matching topic
            
            # Check for preferred entities (ICD codes, etc.)
            doc_entities = doc.metadata.get("entities", [])
            preferred_entities = filters.get("preferred_entities", [])
            
            if preferred_entities:
                entity_overlap = len(set(doc_entities) & set(preferred_entities))
                if entity_overlap > 0:
                    boost_factor *= (1.0 + 0.3 * entity_overlap)  # +30% per matching entity
            
            # Check for rules/thresholds (critical for fraud validation)
            if filters.get("must_contain_rules", False):
                if doc.metadata.get("contains_rule", False):
                    boost_factor *= 1.5  # +50% if contains actionable rules
            
            # Apply boost to score
            boosted_score = score * boost_factor
            
            filtered.append((doc, boosted_score))
        
        return filtered

    # ========== MULTI-STAGE RETRIEVAL ==========
    
    def retrieve_with_filters(
        self,
        query: str,
        fraud_case: Optional[Dict] = None,
        fraud_type: Optional[str] = None,
        top_k: Optional[int] = None,
        priority_topics: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Hybrid retrieval with multi-stage ranking:
        1. Semantic search (vector similarity)
        2. Metadata filtering and boosting
        3. Document type priority
        4. Final re-ranking and deduplication
        """
        
        if self.vector_store is None:
            print("⚠️  No vector store available — using mock retrieval")
            return self._mock_retrieval(query, fraud_type)

        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL

        # Stage 1: Initial semantic search (retrieve 3x for filtering)
        print(f"  🔍 Semantic search: retrieving {top_k } candidates...")
        
        try:
            # ========== FIX: Use similarity_search instead of similarity_search_with_score ==========
            # Supabase doesn't support similarity_search_with_score, use similarity_search + manual scoring
            raw_docs = self.vector_store.similarity_search(
                query, 
                k=top_k 
            )
            
            # Manual scoring based on retrieval order (first = most relevant)
            # Score: 1.0 (most relevant) to 0.5 (least relevant in top-k*3)
            raw_results = []
            for i, doc in enumerate(raw_docs):
                # Linear decay scoring
                score = 1.0 - (i / (len(raw_docs) * 2))  # Scores from 1.0 to 0.5
                raw_results.append((doc, score))
            # ========================================================================================
            
            print(f"  📊 Retrieved {len(raw_results)} documents")
            
        except Exception as e:
            print(f"  ⚠️  Search failed: {e}")
            return self._mock_retrieval(query, fraud_type)

        # Stage 2: Apply similarity threshold
        filtered_results = [
            (doc, score) for doc, score in raw_results 
            if score >= self.config.SIMILARITY_THRESHOLD
        ]
        
        print(f"  ✂️  After similarity threshold ({self.config.SIMILARITY_THRESHOLD}): {len(filtered_results)} docs")

        # Stage 3: Metadata-based filtering and boosting
        if fraud_case:
            metadata_filters = self._extract_metadata_filters(fraud_case)
            filtered_results = self._apply_metadata_filters(filtered_results, metadata_filters)
            print(f"  🏷️  Applied metadata filters: {len(filtered_results)} docs")

        # Stage 4: Document type priority boosting
        priority_boosted = []
        
        for doc, score in filtered_results:
            doc_type = doc.metadata.get("document_type", "")
            doc_priority = doc.metadata.get("priority", "medium")
            
            type_boost = 1.0
            
            # Priority by document type (tariff reference is KING for billing fraud)
            if doc_type == "tariff_reference":
                type_boost *= 1.5
            elif doc_type == "regulation":
                type_boost *= 1.3
            elif doc_type == "guideline":
                type_boost *= 1.2
            
            # Priority by document importance
            priority_multiplier = {
                "critical": 1.3,
                "high": 1.15,
                "medium": 1.0,
                "low": 0.9
            }
            type_boost *= priority_multiplier.get(doc_priority, 1.0)
            
            boosted_score = score * type_boost
            priority_boosted.append((doc, boosted_score))

        # Stage 5: Final ranking and deduplication
        priority_boosted.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplication by content similarity (avoid near-duplicate chunks)
        final_docs = []
        seen_content = set()
        
        for doc, score in priority_boosted:
            # Create content fingerprint (first 200 chars)
            fingerprint = doc.page_content[:200].strip()
            
            if fingerprint not in seen_content:
                final_docs.append(doc)
                seen_content.add(fingerprint)
            
            if len(final_docs) >= top_k:
                break
        
        print(f"  ✅ Final result: {len(final_docs)} documents")
        
        # Log retrieved document types
        from collections import Counter
        doc_types = Counter([d.metadata.get("document_type", "unknown") for d in final_docs])
        print(f"  📚 Document types: {dict(doc_types)}")
        
        return final_docs


    # ========== MOCK RETRIEVAL FOR TESTING ==========
    
    def _mock_retrieval(self, query: str, fraud_type: Optional[str] = None) -> List[Document]:
        """Mock retrieval for development without Supabase"""
        
        fraud_type = fraud_type or "general"
        
        mock_docs = [
            Document(
                page_content=f"[MOCK] Permenkes INA-CBG: Untuk kasus {fraud_type}, tarif standar Kelas III adalah... Markup tidak boleh melebihi 10% dari tarif dasar INA-CBG.",
                metadata={
                    "title": "INA-CBG Tariff Reference",
                    "document_type": "tariff_reference",
                    "priority": "critical",
                    "key_topics": ["tariff_validation", "inacbg_rates"],
                    "entities": ["M-1-50-I", "Kelas III"],
                    "contains_rule": True,
                }
            ),
            Document(
                page_content=f"[MOCK] PerBPJS No. 6/2020 Pasal 15: Fraud tipe {fraud_type} dikenakan sanksi administratif berupa penundaan pembayaran klaim. Faskes wajib mengembalikan selisih klaim yang tidak wajar.",
                metadata={
                    "title": "PerBPJS 6/2020 - Anti-Fraud",
                    "document_type": "regulation",
                    "priority": "high",
                    "key_topics": ["fraud_penalties", "investigation_procedures"],
                    "entities": ["Pasal 15"],
                    "contains_rule": True,
                }
            ),
            Document(
                page_content=f"[MOCK] Clinical Pathway: Untuk diagnosis pneumonia (J18.9), typical LOS adalah 5 hari. Perawatan lebih dari 7 hari memerlukan justifikasi medis.",
                metadata={
                    "title": "Clinical Practice Guidelines",
                    "document_type": "guideline",
                    "priority": "high",
                    "key_topics": ["clinical_pathways", "length_of_stay"],
                    "entities": ["J18.9", "pneumonia"],
                    "contains_rule": True,
                }
            ),
        ]
        
        return mock_docs
