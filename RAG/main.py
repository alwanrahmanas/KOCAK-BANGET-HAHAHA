"""
BPJS Fraud Detection - RAG System Main Entry Point
==================================================
Orchestrates all RAG components with clean CLI interface
Loads configuration from .env file
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env
load_dotenv(override=True)
from utils.env_loader import get_env

API_KEY = get_env("API_KEY")
DB_URL = get_env("DB_URL")

# Import all RAG system components
from rag_config import RAGConfig
from document_chunker import DocumentChunker
from explanation_generator import ExplanationGenerator
from ml_pipeline_rag_bridge import MLPipelineRAGBridge
from retrieval_engine import RetrievalEngine
from vector_store_manager import VectorStoreManager
from bpjs_fraud_rag_system import BPJSFraudRAGSystem
import logging
# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def print_banner():
    """Print welcome banner"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              BPJS FRAUD DETECTION - RAG SYSTEM v1.0                      ║
║              Explainable AI dengan Regulasi & INA-CBG                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)


def check_environment():
    """Check if all required environment variables are set"""
    print("\n" + "="*80)
    print("CHECKING ENVIRONMENT")
    print("="*80)
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'SUPABASE_URL': 'Supabase URL',
        'SUPABASE_KEY': 'Supabase Anon Key'
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var:
                display_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            else:
                display_value = value[:30] + "..." if len(value) > 30 else value
            
            print(f"✓ {description}: {display_value}")
        else:
            print(f"✗ {description}: NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  WARNING: Missing environment variables: {', '.join(missing_vars)}")
        print("   System will run in MOCK mode (no actual vector store)")
        print("   Create .env file with:")
        for var in missing_vars:
            print(f"   {var}=your-value-here")
        return False
    else:
        print("\n✅ All environment variables configured!")
        return True


def check_documents():
    """Check if required documents are available"""
    print("\n" + "="*80)
    print("CHECKING DOCUMENTS")
    print("="*80)
    
    # Update these paths based on your actual file names
    document_paths = {
        'PerBPJS No. 6/2020': r'C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\Peraturan BPJS Kesehatan No. 6 Tahun 2020 tentang Sistem Pencegahan Kecurangan Dalam Pelaksanaan Program Jaminan Kesehatan.pdf',
        'PerBPJS No. 5/2020': r'C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\Peraturan BPJS Kesehatan Nomor 5 Tahun 2020.pdf',
        'INA-CBG': r'C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\INA-CBGs.pdf',
        'Pedoman INA-CBGs': r"C:\Users\US3R\OneDrive\Dokumen\Referensi\RAG-Bahan\Pedoman INACBGs.pdf"
    }
    
    found_count = 0
    
    for doc_name, doc_path in document_paths.items():
        # Check if file exists in current directory
        if os.path.exists(doc_path):
            file_size = os.path.getsize(doc_path)
            print(f"✓ {doc_name}: {doc_path} ({file_size:,} bytes)")
            found_count += 1
        else:
            print(f"✗ {doc_name}: {doc_path} (NOT FOUND)")
    
    print(f"\n📄 Documents found: {found_count}/{len(document_paths)}")
    
    if found_count == 0:
        print("⚠️  No documents found. Place PDF files in the same directory.")
        return False
    
    return True


def cmd_setup(args):
    """Setup RAG system: load documents and build vector store"""
    print("\n" + "="*80)
    print("COMMAND: SETUP RAG SYSTEM")
    print("="*80)
    
    print(f"\nMode: {'REBUILD' if args.rebuild else 'LOAD EXISTING'}")
    
    # Initialize RAG system
    print("\n[1/2] Initializing RAG system...")
    rag_system = BPJSFraudRAGSystem()
    
    # Setup with or without rebuild
    print(f"\n[2/2] {'Building' if args.rebuild else 'Loading'} vector store...")
    rag_system.setup(rebuild_vectors=args.rebuild)
    
    print("\n" + "="*80)
    print("✅ SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  • Test retrieval: python main.py test")
    print("  • Generate explanation: python main.py explain")


def cmd_test(args):
    """Test RAG system with sample queries"""
    print("\n" + "="*80)
    print("COMMAND: TEST RAG SYSTEM")
    print("="*80)
    
    # Initialize RAG system
    print("\n[1/2] Loading RAG system...")
    rag_system = BPJSFraudRAGSystem()
    rag_system.setup(rebuild_vectors=False)
    
    # Test cases
    test_cases = [
        {
            'name': 'Inflated Bill - High Claim Ratio',
            'fraud_case': {
                'claim_id': 'TEST_001',
                'predicted_fraud': 1,
                'fraud_probability': 0.94,
                'predicted_fraud_type': 'inflated_bill',
                'top_features': 'claim_ratio↑, selisih_klaim↑, lama_dirawat↓',
                'explanation_summary': 'Claim ratio 2.8x normal, selisih 3M',
                'explanation_json': {
                    'top_features': [
                        {'feature': 'claim_ratio', 'value': 2.8, 'vs_mean': '2.8x avg', 
                         'z_score': 4.5, 'impact': 'strong_positive', 'direction': 'up'},
                        {'feature': 'selisih_klaim', 'value': 3000000, 'vs_mean': '3x avg',
                         'z_score': 3.2, 'impact': 'strong_positive', 'direction': 'up'}
                    ]
                },
                'claim_data': {
                    'diagnosis': 'A09.0 - Gastroenteritis Acute',
                    'procedure': '87.03 - Radiografi thorax',
                    'jenis_pelayanan': 'Rawat Inap',
                    'room_class': 'Kelas 1',
                    'lama_dirawat': 1,
                    'billed_amount': 8000000,
                    'paid_amount': 8000000,
                    'tarif_inacbg': 5000000,
                    'selisih_klaim': 3000000,
                    'claim_ratio': 1.6,
                    'drug_cost': 2000000,
                    'procedure_cost': 4000000
                }
            }
        }
    ]
    
    if args.case_id:
        # Filter specific test case
        test_cases = [tc for tc in test_cases if tc['fraud_case']['claim_id'] == args.case_id]
        if not test_cases:
            print(f"✗ Test case {args.case_id} not found")
            return
    
    # Run tests
    print(f"\n[2/2] Running {len(test_cases)} test case(s)...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'='*80}")
        
        fraud_case = test_case['fraud_case']
        print(f"\nClaim ID: {fraud_case['claim_id']}")
        print(f"Fraud Type: {fraud_case['predicted_fraud_type']}")
        print(f"Probability: {fraud_case['fraud_probability']:.1%}")
        print(f"Top Features: {fraud_case['top_features']}")
        
        # Generate explanation
        try:
            result = rag_system.explain_fraud_case(fraud_case)
            
            # print(f"\n{'─'*80}")
            # print("EXPLANATION GENERATED")
            # print(f"{'─'*80}")
            # print(f"Tokens used: {result['tokens_used']}")
            # print(f"Cost: ${result['cost']:.4f}")
            # print(f"Retrieved docs: {len(result['retrieved_docs'])}")
            
            print(f"\n{'─'*80}")
            print("EXPLANATION TEXT")
            print(f"{'─'*80}")
            print(result['explanation_text'])
            
            # Save to file if requested
            if args.save:
                output_path = f"./output/test_{fraud_case['claim_id']}.txt"
                os.makedirs('./output', exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Test Case: {test_case['name']}\n")
                    f.write(f"Claim ID: {fraud_case['claim_id']}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(result['explanation_text'])
                print(f"\n💾 Saved to: {output_path}")
        
        except Exception as e:
            print(f"\n✗ Error generating explanation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)

# Version with logger and debugger integration (breakpoints), without changing logic.

# import os
# import pandas as pd
# import json
# import logging

# # Setup logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# # Console handler
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)
# claim_data = {}
# for source, target in colmap.items():
#     claim_data[target] = row.get(source)
def cmd_explain(args):
    """Generate explanation for fraud cases from CSV (FULLY FIXED VERSION)"""

    logger.info("Starting cmd_explain execution")

    print("\n" + "="*80)
    print("COMMAND: GENERATE EXPLANATIONS")
    print("="*80)

    # Check input file
    if not args.input:
        logger.error("--input file missing")
        print("✗ Error: --input file is required")
        return

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        print(f"✗ Error: Input file not found: {args.input}")
        return

    # Load predictions
    logger.debug(f"Loading predictions CSV: {args.input}")
    print(f"\n[1/4] Loading predictions from: {args.input}")
    predictions_df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(predictions_df)} predictions from CSV")

    # Print available columns for debugging
    print(f"\n[Debug] Available columns in CSV ({len(predictions_df.columns)}):")
    print(f"   {', '.join(predictions_df.columns[:20])}...")

    fraud_df = predictions_df[predictions_df['predicted_fraud'] == 1].copy()
    logger.info(f"Detected {len(fraud_df)} fraud cases")

    print(f"   Loaded {len(predictions_df)} predictions")
    print(f"   Fraud cases detected: {len(fraud_df)}")

    if len(fraud_df) == 0:
        logger.warning("No fraud cases found; stopping execution")
        print("   No fraud cases to explain. Exiting.")
        return

    # Optional RAG data merge
    if hasattr(args, 'rag_data') and args.rag_data:
        logger.debug(f"rag_data argument detected: {args.rag_data}")

        if os.path.exists(args.rag_data):
            print(f"\n[2/4] Loading comprehensive RAG data from: {args.rag_data}")
            rag_df = pd.read_csv(args.rag_data)
            logger.info(f"Loaded {len(rag_df)} rows of RAG enrichment data")

            fraud_rag_df = fraud_df.merge(rag_df, on='claim_id', suffixes=('_pred', '_rag'))
            logger.info(f"Merged dataset resulting rows: {len(fraud_rag_df)}")
            print(f"   Merged fraud and RAG data rows: {len(fraud_rag_df)}")
        else:
            logger.error(f"RAG file not found: {args.rag_data}")
            print(f"✗ Error: RAG data file not found: {args.rag_data}")
            return
    else:
        logger.debug("No rag_data argument provided; using fraud_df only")
        fraud_rag_df = fraud_df

    # Initialize RAG system
    logger.debug("Initializing RAG system")
    print("\n[3/4] Loading RAG system...")
    rag_system = BPJSFraudRAGSystem()
    rag_system.setup(rebuild_vectors=False)

    # Prepare fraud cases
    logger.debug("Preparing fraud cases for RAG explanation")
    print("\n[4/4] Preparing fraud cases for explanation...")
    
    fraud_cases = []

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL FIX: Use ORIGINAL CSV column names (no mapping needed!)
    # ExplanationGenerator already expects these exact names
    # ═══════════════════════════════════════════════════════════════
    
    # List of columns to pass through (use CSV column names as-is)
    columns_to_pass = [
        # Core identifiers
        'claim_id',
        
        # Fraud prediction
        'predicted_fraud',
        'fraud_probability', 
        'predicted_fraud_type',
        'fraud_label',
        
        # SHAP explanation
        'shap_explanation_summary',
        'shap_top_features',
        
        # Clinical data - EXACT CSV column names
        'kode_icd10',           # ✓ Keep original name
        'diagnosis_name',       # ✓ Keep original name  
        'clinical_pathway_name', # ✓ Keep original name
        'kode_prosedur',        # ✓ Keep original name
        'procedure_name',       # ✓ Keep original name
        
        # Financial data
        'inacbg_code',
        'kode_tarif_inacbg',
        'tarif_inacbg',
        'inacbg_tarif',
        'billed_amount',
        'paid_amount',
        'selisih_klaim',
        'claim_ratio',
        
        # Ratio indicators
        'drug_ratio',
        'procedure_ratio',
        
        # Service data
        'jenis_pelayanan',
        'room_class',
        'lama_dirawat',
        
        # Provider data
        'faskes_id',
        'faskes_name',  # May not exist
        'faskes_level',
        'provider_monthly_claims',
        
        # Patient history
        'visit_count_30d',
        'clinical_pathway_deviation_score',
        
        # Additional context
        'drug_cost',
        'procedure_cost',
        'obat_keluar',
        'obat_match_score',
    ]

    for idx, row in fraud_rag_df.iterrows():
        logger.debug(f"Processing claim_id: {row.get('claim_id')}")

        # Parse explanation_json if string
        explanation_json = row.get('explanation_json', {})
        if isinstance(explanation_json, str):
            try:
                explanation_json = json.loads(explanation_json)
            except Exception as e:
                logger.warning(f"Failed to parse explanation_json for {row.get('claim_id')}: {e}")
                explanation_json = {}

        # ═══════════════════════════════════════════════════════════════
        # FIXED: Build fraud_case with ORIGINAL column names
        # No mapping needed - pass CSV columns directly
        # ═══════════════════════════════════════════════════════════════
        fraud_case = {}
        
        # Copy all relevant columns with their ORIGINAL names
        for col in columns_to_pass:
            if col in row.index:
                fraud_case[col] = row[col]
            else:
                # If column doesn't exist, set to None
                fraud_case[col] = None

        # Add explanation_json
        fraud_case['explanation_json'] = explanation_json
        
        # Ensure fraud_label exists
        if 'fraud_label' not in fraud_case or pd.isna(fraud_case['fraud_label']):
            fraud_case['fraud_label'] = "FRAUD" if fraud_case.get('predicted_fraud') == 1 else "LEGITIMATE"

        fraud_cases.append(fraud_case)

    logger.info(f"Total fraud cases prepared: {len(fraud_cases)}")

    # ═══════════════════════════════════════════════════════════════
    # DIAGNOSTIC: Print first case structure
    # ═══════════════════════════════════════════════════════════════
    if fraud_cases:
        print("\n" + "="*80)
        print("[DIAGNOSTIC] First fraud case structure:")
        print("="*80)
        
        first_case = fraud_cases[0]
        
        # Check critical fields
        critical_fields = {
            'claim_id': first_case.get('claim_id'),
            'kode_icd10': first_case.get('kode_icd10'),
            'diagnosis_name': first_case.get('diagnosis_name'),
            'tarif_inacbg': first_case.get('tarif_inacbg'),
            'billed_amount': first_case.get('billed_amount'),
            'paid_amount': first_case.get('paid_amount'),
            'selisih_klaim': first_case.get('selisih_klaim'),
            'claim_ratio': first_case.get('claim_ratio'),
        }
        
        print("\nCritical Fields:")
        for field, value in critical_fields.items():
            status = "✓" if value is not None and str(value).strip() != '' else "✗ NULL"
            print(f"   {status} {field}: {value}")
        
        # Check SHAP features
        shap_features = first_case.get('shap_top_features')
        print(f"\nSHAP Features Type: {type(shap_features)}")
        if isinstance(shap_features, str):
            print(f"   First 100 chars: {shap_features[:100]}")
        
        print("\n" + "="*80)

    # Call RAG batch explanation
    logger.debug("Calling RAG batch explanation engine")
    print("\n[5/5] Generating RAG explanations...")
    
    rag_explanations_df = rag_system.explain_fraud_cases_batch(fraud_cases)
    logger.info("RAG explanation batch completed")

    # Set output path
    if not args.output:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f'./output/rag_explanations_{timestamp}.csv'
        logger.debug(f"Output path auto-generated: {args.output}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Merge explanations back
    logger.debug("Merging RAG explanations back to full prediction dataframe")
    final_df = MLPipelineRAGBridge.merge_rag_explanations_back(predictions_df, rag_explanations_df)
    final_df.to_csv(args.output, index=False)

    logger.info(f"Saved explanation output to: {args.output}")
    logger.info(f"Total rows in final output: {len(final_df)}")

    print("\n" + "="*80)
    print("✅ EXPLANATIONS COMPLETE!")
    print("="*80)
    print(f"📄 Output saved to: {args.output}")
    print(f"   Total rows: {len(final_df)}")
    print(f"   Fraud cases explained: {len(rag_explanations_df)}")


def cmd_info(args):
    """Show system information"""
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    
    print("\n📦 Python Packages:")
    packages = ['langchain', 'openai', 'supabase', 'pandas', 'numpy']
    for pkg in packages:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✓ {pkg}: {version}")
        except ImportError:
            print(f"   ✗ {pkg}: NOT INSTALLED")
    
    print("\n🔧 Configuration:")
    config = RAGConfig()
    print(f"   Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"   LLM Model: {config.LLM_MODEL}")
    print(f"   Chunk Size: {config.CHUNK_SIZE}")
    print(f"   Top-K Retrieval: {config.TOP_K_RETRIEVAL}")
    print(f"   Vector Table: {config.VECTOR_TABLE_NAME}")
    
    print("\n📊 File Structure:")
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    print(f"   Python modules: {len(py_files)}")
    for f in sorted(py_files):
        print(f"     • {f}")
    
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    print(f"\n   PDF documents: {len(pdf_files)}")
    for f in sorted(pdf_files):
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"     • {f} ({size_mb:.2f} MB)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BPJS Fraud Detection RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time setup - build vector store
  python main.py setup --rebuild
  
  # Load existing vector store
  python main.py setup
  
  # Test with sample cases
  python main.py test
  
  # Generate explanations from predictions CSV
  python main.py explain --input predictions.csv --output results.csv
  
  # Show system info
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup RAG system')
    setup_parser.add_argument('--rebuild', action='store_true',
                             help='Rebuild vector store from scratch')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test RAG system')
    test_parser.add_argument('--case-id', type=str,
                            help='Specific test case ID to run')
    test_parser.add_argument('--save', action='store_true',
                            help='Save results to file')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', 
                                          help='Generate explanations from predictions')
    explain_parser.add_argument('--input', type=str, required=True,
                               help='Input CSV with predictions')
    explain_parser.add_argument('--output', type=str,
                               help='Output CSV path (default: ./output/fraud_explanations.csv)')
    explain_parser.add_argument('--limit', type=int,
                               help='Limit number of cases to process')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check environment
    env_ok = check_environment()
    doc_ok = check_documents()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to command handlers
    if args.command == 'setup':
        cmd_setup(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'explain':
        cmd_explain(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
