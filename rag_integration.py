"""
RAG Integration Module for ML Inference Pipeline
Handles connection between ML predictions and RAG explanation system
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging
# ⭐ ADD THIS - Load .env file
from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)


class RAGIntegration:
    """Manages RAG system integration with ML pipeline"""
    
    def __init__(self, rag_system_path: Optional[str] = None):
        """
        Initialize RAG integration
        
        Args:
            rag_system_path: Path to RAG system directory
                            If None, will try from env or default location
        """
        self.rag_system_path = rag_system_path or os.getenv('RAG_SYSTEM_PATH')
        
        if not self.rag_system_path:
            # Try default path (one level up)
            default_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'RAG-System')
            )
            if os.path.exists(default_path):
                self.rag_system_path = default_path
        
        self.rag_available = False
        self.rag_system = None
        self.bridge = None
        
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if RAG system is available"""
        if not self.rag_system_path:
            logger.warning("RAG system path not configured")
            return False
        
        if not os.path.exists(self.rag_system_path):
            logger.warning(f"RAG system path not found: {self.rag_system_path}")
            return False
        
        try:
            # Add RAG system to path
            if self.rag_system_path not in sys.path:
                sys.path.insert(0, self.rag_system_path)
            
            # Try importing RAG modules
            from bpjs_fraud_rag_system import BPJSFraudRAGSystem
            from ml_pipeline_rag_bridge import MLPipelineRAGBridge
            
            self.BPJSFraudRAGSystem = BPJSFraudRAGSystem
            self.MLPipelineRAGBridge = MLPipelineRAGBridge
            
            self.rag_available = True
            logger.info(f"✓ RAG system available at: {self.rag_system_path}")
            return True
            
        except ImportError as e:
            logger.warning(f"RAG system import failed: {e}")
            self.rag_available = False
            return False
    
    def initialize_rag_system(self, rebuild_vectors: bool = False):
        """Initialize and setup RAG system"""
        if not self.rag_available:
            raise RuntimeError("RAG system not available")
        
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING RAG SYSTEM")
        logger.info("="*80)
        
        # Initialize RAG system
        logger.info("\n[1/2] Creating RAG system instance...")
        self.rag_system = self.BPJSFraudRAGSystem()
        
        # Setup (load vector store)
        logger.info(f"[2/2] {'Rebuilding' if rebuild_vectors else 'Loading'} vector store...")
        self.rag_system.setup(rebuild_vectors=rebuild_vectors)
        
        logger.info("✓ RAG system initialized successfully\n")
        
        return self.rag_system
    
    def generate_explanations(
        self,
        predictions_df: pd.DataFrame,
        original_claims_df: pd.DataFrame,
        fraud_only: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Generate RAG explanations for fraud cases
        
        Args:
            predictions_df: ML predictions DataFrame
            original_claims_df: Original claims DataFrame
            fraud_only: Only explain fraud cases (True) or all (False)
            show_progress: Show progress during generation
        
        Returns:
            DataFrame with RAG explanations
        """
        if not self.rag_available:
            raise RuntimeError("RAG system not available")
        
        if self.rag_system is None:
            self.initialize_rag_system()
        
        logger.info("\n" + "="*80)
        logger.info("GENERATING RAG EXPLANATIONS")
        logger.info("="*80)
        
        # Prepare fraud cases
        logger.info("\n[1/3] Preparing fraud cases for RAG...")
        fraud_cases = self.MLPipelineRAGBridge.prepare_fraud_cases_for_rag(
            predictions_df=predictions_df,
            original_claims_df=original_claims_df,
            fraud_only=fraud_only
        )
        
        logger.info(f"   Found {len(fraud_cases)} cases to explain")
        
        if len(fraud_cases) == 0:
            logger.info("   No cases to explain")
            return pd.DataFrame()
        
        # Generate explanations
        logger.info(f"\n[2/3] Generating explanations...")
        logger.info(f"   Model: GPT-4o-mini")
        logger.info(f"   Estimated time: ~{len(fraud_cases) * 3} seconds")
        
        rag_explanations_df = self.rag_system.explain_fraud_cases_batch(fraud_cases)
        
        logger.info(f"   ✓ Generated {len(rag_explanations_df)} explanations")
        
        # Merge back
        logger.info(f"\n[3/3] Merging with predictions...")
        final_df = self.MLPipelineRAGBridge.merge_rag_explanations_back(
            predictions_df=predictions_df,
            rag_explanations_df=rag_explanations_df
        )
        
        logger.info(f"   ✓ Merge complete")
        logger.info(f"   Final columns: {len(final_df.columns)}")
        
        return final_df
    
    def is_available(self) -> bool:
        """Check if RAG system is available"""
        return self.rag_available
    
    def get_status(self) -> Dict:
        """Get RAG integration status"""
        return {
            'available': self.rag_available,
            'path': self.rag_system_path,
            'initialized': self.rag_system is not None,
        }
if __name__ == "__main__":
    # Test RAG Integration
    rag_integration = RAGIntegration()
    if rag_integration.is_available():
        rag_system = rag_integration.initialize_rag_system(rebuild_vectors=False)
        logger.info("RAG Integration test successful")
    else:
        logger.error("RAG Integration test failed: RAG system not available")