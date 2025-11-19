# ==================== VECTOR STORE MANAGER ====================
from typing import List
from rag_config import RAGConfig

# --- LangChain New Modular Packages ----
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

# --- Supabase Client ---
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase import create_client, Client
from rag_config import RAGConfig

class VectorStoreManager:
    """Manage Supabase vector store for document retrieval"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.supabase_client: Client | None = None
        self.embeddings = None
        self.vector_store = None

    # ----------------------------------------------------------------------
    def initialize(self):
        """Initialize OpenAI embeddings + Supabase client"""
        print("\n" + "=" * 80)
        print("INITIALIZING VECTOR STORE")
        print("=" * 80)

        # Embeddings
        print("\n[Embeddings] Initializing OpenAI embeddings...")
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                api_key=self.config.OPENAI_API_KEY
            )
            print(f"  ✓ Using model: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"  ✗ Failed to initialize embeddings: {e}")
            return

        # Supabase Client
        print("\n[Supabase] Connecting...")
        try:
            self.supabase_client = create_client(
                self.config.SUPABASE_URL,
                self.config.SUPABASE_KEY
            )
            print("  ✓ Connected to Supabase")
        except Exception as e:
            print(f"  ✗ Could not connect to Supabase: {e}")
            self.supabase_client = None

        print("\n✅ Vector store initialized")

    # ----------------------------------------------------------------------
    def create_vector_store(self, documents: List[Document]) -> Optional[SupabaseVectorStore]:
   
        if self.supabase_client is None:
            print("⚠️ Supabase client not initialized. Running in MOCK mode.")
            return None
        
        print("\n" + "="*80)
        print("CREATING VECTOR STORE")
        print("="*80)
        
        try:
            print(f"  📊 Processing {len(documents)} document chunks...")
            
            # Calculate embeddings cost estimate
            total_chars = sum(len(doc.page_content) for doc in documents)
            estimated_tokens = total_chars // 4
            estimated_cost = (estimated_tokens / 1000) * 0.00002  # $0.02 per 1M tokens
            
            print(f"  💰 Estimated cost: ${estimated_cost:.4f}")
            print(f"  📝 Total tokens: ~{estimated_tokens:,}")
            
            # Create vector store WITHOUT ids parameter (let Supabase auto-generate UUID)
            print(f"  🔄 Generating embeddings and inserting to Supabase...")
            
            vector_store = SupabaseVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.supabase_client,
                table_name=self.config.VECTOR_TABLE_NAME,
                query_name="match_documents",
                chunk_size=500  # Batch insert to avoid timeout
                # ← JANGAN ADA ids parameter di sini!
            )
            
            print(f"  ✅ Vector store created!")
            print(f"     Table: {self.config.VECTOR_TABLE_NAME}")
            print(f"     Vectors inserted: {len(documents)}")
            
            return vector_store
        
        except Exception as e:
            print(f"  ✗ Error creating vector store: {e}")
            import traceback
            traceback.print_exc()
            return None


    # ----------------------------------------------------------------------
    def load_existing_vector_store(self):
        """Load existing vector store from Supabase"""
        print("\n" + "=" * 80)
        print("LOADING EXISTING VECTOR STORE")
        print("=" * 80)

        if not self.supabase_client:
            print("⚠️  Cannot load vector store (Supabase client not initialized)")
            return None

        try:
            self.vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name=self.config.VECTOR_TABLE_NAME,
                query_name="match_documents"
            )
            print("✅ Vector store loaded successfully")
            return self.vector_store

        except Exception as e:
            print(f"✗ Error loading vector store: {e}")
            return None
