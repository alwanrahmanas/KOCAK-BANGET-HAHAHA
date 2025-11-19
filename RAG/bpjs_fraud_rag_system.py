# ==================== MAIN RAG SYSTEM ====================
import pandas as pd
from typing import List, Dict

from rag_config import RAGConfig
from document_loader import DocumentLoader
from document_chunker import DocumentChunker
from vector_store_manager import VectorStoreManager
from retrieval_engine import RetrievalEngine
from explanation_generator import ExplanationGenerator


class BPJSFraudRAGSystem:
    """Complete RAG system for BPJS fraud explanation."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

        self.document_loader = DocumentLoader(self.config)
        self.chunker = DocumentChunker(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)

        self.retrieval_engine = None
        self.explanation_generator = None

    # ------------------------------------------------------------------

    def setup(self, rebuild_vectors: bool = False):
        """Setup RAG system: load documents, build vectors, init engines."""

        print("\n" + "="*80)
        print("BPJS FRAUD RAG SYSTEM - SETUP")
        print("="*80)

        # Initialize client & embeddings
        self.vector_store_manager.initialize()

        if rebuild_vectors:
            docs = self.document_loader.load_all_documents()
            chunks = self.chunker.chunk_all_documents(docs)
            vector_store = self.vector_store_manager.create_vector_store(chunks)
        else:
            vector_store = self.vector_store_manager.load_existing_vector_store()

        # Build engine
        if vector_store is None:
            print("⚠️ Running in MOCK mode — vector store unavailable")
            self.retrieval_engine = RetrievalEngine(None, self.config)
        else:
            self.retrieval_engine = RetrievalEngine(vector_store, self.config)

        # Attach LLM explanation system
        self.explanation_generator = ExplanationGenerator(
            self.config,
            self.retrieval_engine
        )

        print("\n✅ RAG System Ready.\n")

    # ------------------------------------------------------------------

    def explain_fraud_case(self, fraud_case: Dict) -> Dict:
        if self.explanation_generator is None:
            raise RuntimeError("System not initialized. Call setup().")
        return self.explanation_generator.generate_explanation(fraud_case)

    # ------------------------------------------------------------------

    def explain_fraud_cases_batch(self, fraud_cases: List[Dict]) -> pd.DataFrame:
        if self.explanation_generator is None:
            raise RuntimeError("System not initialized. Call setup().")
        return self.explanation_generator.batch_generate_explanations(fraud_cases)
