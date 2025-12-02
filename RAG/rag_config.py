# ==================== CONFIGURATION ====================
import os
# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(override=True)

class RAGConfig:
    """Central configuration for RAG system"""
    
    # API Keys (Load from environment)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # Document Paths
    DOCUMENT_PATHS = {
        'perbpjs_6_2020': r'C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\Peraturan BPJS Kesehatan No. 6 Tahun 2020 tentang Sistem Pencegahan Kecurangan Dalam Pelaksanaan Program Jaminan Kesehatan.pdf',
        'perbpjs_5_2020': r'C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\Peraturan BPJS Kesehatan Nomor 5 Tahun 2020.pdf',
        'inacbg': r'C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\INA-CBGs.pdf',
        'pedoman_inacbg': r"C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\Pedoman INACBGs.pdf"
    }
    
    # Document Metadata (Base metadata per document)
    DOCUMENT_METADATA = {
        'perbpjs_6_2020': {
            'title': 'PerBPJS No. 6/2020 - Anti-Fraud Regulations',
            'document_type': 'regulation',
            'issuer': 'BPJS Kesehatan',
            'year': 2020,
            'priority': 'high',
            'base_topics': ['fraud_detection', 'penalties', 'investigation_procedures', 'fraud_prevention']
        },
        'perbpjs_5_2020': {
            'title': 'PerBPJS No. 5/2020 - Claims Management',
            'document_type': 'regulation',
            'issuer': 'BPJS Kesehatan',
            'year': 2020,
            'priority': 'high',
            'base_topics': ['claims_procedure', 'verification', 'payment_standards']
        },
        'inacbg': {
            'title': 'INA-CBG Tariff Reference',
            'document_type': 'tariff_reference',
            'issuer': 'Kementerian Kesehatan',
            'year': 2023,
            'priority': 'critical',  # ⭐ MOST IMPORTANT for tariff validation
            'base_topics': ['tariff_validation', 'inacbg_codes', 'billing_standards', 'diagnosis_groups']
        },
        'pedoman_inacbg': {
            'title': 'Pedoman INA-CBG',
            'document_type': 'guideline',
            'issuer': 'Kementerian Kesehatan',
            'year': 2023,
            'priority': 'critical',
            'base_topics': ['inacbg_implementation', 'coding_guidelines', 'tariff_calculation', 'clinical_pathways']
        }
    }
    
    # Model Configuration
    EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cost-effective
    LLM_MODEL = "gpt-4.1"  # Fast and cheap for metadata generation
    LLM_TEMPERATURE = 0.5  # Low temperature for factual responses
    
    # Chunking Configuration (Optimized for legal/medical documents)
    # Research shows 500-800 tokens optimal for semantic retrieval
    # With avg 4 chars/token, 500 tokens ≈ 2000 chars, 800 tokens ≈ 3200 chars
    CHUNK_SIZE = 2500              # Characters (≈625 tokens) - optimal for regulation paragraphs
    CHUNK_OVERLAP = 500            # 20% overlap to preserve context across boundaries
    
    # Alternative chunk sizes for different document types
    CHUNK_SIZES = {
        'regulation': 2500,        # Longer for legal text with numbered articles
        'tariff_reference': 1500,  # Shorter for structured tariff tables
        'guideline': 2000,         # Medium for procedural guidelines
        'clinical': 2000           # Medium for clinical pathways
    }
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Supabase Table
    VECTOR_TABLE_NAME = "bpjs_fraud_regulations"
    
    # LLM Metadata Generation
    ENABLE_LLM_METADATA = True     # Set False to skip (faster but less rich metadata)
    METADATA_BATCH_SIZE = 10       # Process N chunks before calling LLM (cost optimization)
