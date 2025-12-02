import pandas as pd
from typing import List, Dict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from RAG.rag_config import RAGConfig
from RAG.document_loader import DocumentLoader
from RAG.document_chunker import DocumentChunker
from RAG.vector_store_manager import VectorStoreManager
from RAG.retrieval_engine import RetrievalEngine
from RAG.explanation_generator import ExplanationGenerator
from models.inference_engine import BPJSFraudInferenceEngine


class BPJSFraudRAGSystem:
    """Complete RAG system for BPJS fraud explanation."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

        self.document_loader = DocumentLoader(self.config)
        self.chunker = DocumentChunker(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)

        self.retrieval_engine = None
        self.explanation_generator = None
        
        # Initialize inference engine (WITHOUT loading models yet)
        self.inference_engine = BPJSFraudInferenceEngine()
        print("✓ Inference engine initialized (models not loaded)")

    def setup(self, rebuild_vectors: bool = False):
        """Setup RAG system: load documents, build vectors, init engines."""

        print("\n" + "="*80)
        print("BPJS FRAUD RAG SYSTEM - SETUP")
        print("="*80)

        # Initialize client & embeddings
        self.vector_store_manager.initialize()

        if rebuild_vectors:
            print("\n[1] Loading and chunking documents...")
            docs = self.document_loader.load_all_documents()
            chunks = self.chunker.chunk_all_documents(docs)
            print(f"    ✓ Created {len(chunks)} chunks")
            
            print("\n[2] Building vector store...")
            vector_store = self.vector_store_manager.create_vector_store(chunks)
        else:
            print("\n[1] Loading existing vector store...")
            vector_store = self.vector_store_manager.load_existing_vector_store()

        # Build retrieval engine
        if vector_store is None:
            print("    ⚠️  Running in MOCK mode — vector store unavailable")
            self.retrieval_engine = RetrievalEngine(None, self.config)
        else:
            print("    ✓ Vector store loaded")
            self.retrieval_engine = RetrievalEngine(vector_store, self.config)

        # Attach LLM explanation system
        print("\n[2] Initializing explanation generator...")
        self.explanation_generator = ExplanationGenerator(
            self.config,
            self.retrieval_engine
        )

        print("\n✅ RAG System Ready.\n")

    def explain_fraud_case(self, fraud_case: Dict) -> Dict:
        """Generate explanation for a single fraud case."""
        if self.explanation_generator is None:
            raise RuntimeError("System not initialized. Call setup() first.")
        return self.explanation_generator.generate_explanation(fraud_case)

    def explain_fraud_cases_batch(
    self, 
    fraud_cases_df,  # Bisa DataFrame atau List
    fraud_only: bool = True
    ) -> pd.DataFrame:
        """
        Generate explanations for multiple fraud cases
        
        Args:
            fraud_cases_df: DataFrame or List of fraud predictions
            fraud_only: Only process fraud cases
        
        Returns:
            DataFrame with explanation_text and retrieved_docs columns
        """
        if self.explanation_generator is None:
            raise RuntimeError("System not initialized. Call setup() first.")
        
        # ⚠️ FIX: Handle both DataFrame and List input
        if isinstance(fraud_cases_df, pd.DataFrame):
            if fraud_only:
                fraud_cases = fraud_cases_df[fraud_cases_df['predicted_fraud'] == 1].to_dict('records')
            else:
                fraud_cases = fraud_cases_df.to_dict('records')
        elif isinstance(fraud_cases_df, list):
            # Already a list
            fraud_cases = fraud_cases_df
        else:
            raise TypeError(f"Expected DataFrame or list, got {type(fraud_cases_df)}")
        
        # Call explanation generator
        return self.explanation_generator.batch_generate_explanations(fraud_cases, fraud_only=False)
