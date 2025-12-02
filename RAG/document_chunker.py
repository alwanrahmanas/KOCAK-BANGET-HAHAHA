from RAG.rag_config import RAGConfig
from typing import List, Tuple, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import json
import re
import time


class DocumentChunker:
    """
    Smart document chunker with:
    - Adaptive chunk sizing
    - Progress tracking with tqdm
    - LLM-generated metadata enrichment
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize LLM for metadata generation (if enabled)
        self.llm = None
        if config.ENABLE_LLM_METADATA and config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                openai_api_key=config.OPENAI_API_KEY
            )
    
    # ------------------------------------------------------------------
    # Helper: splitter per document type
    def _get_text_splitter(self, document_type: str) -> RecursiveCharacterTextSplitter:
        chunk_size = self.config.CHUNK_SIZES.get(
            document_type, 
            self.config.CHUNK_SIZE
        )
        separators = [
            "\n\n\n",
            "\n\n",
            "\nPasal ",
            "\nBab ",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            ""
        ]
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=separators
        )
    
    # ------------------------------------------------------------------
    # Helper: robust JSON parsing
    def _parse_json_safely(self, text: str, expected_items: int) -> List[Dict]:
        """
        Try to parse JSON array from LLM response robustly.
        - Strips markdown fences
        - Extracts first [...] block
        - Falls back to empty metadata if fails
        """
        text = text.strip()
        
        # # Remove `````` fences if present
        # if text.startswith("```
        #     text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        #     if text.endswith("```
        #         text = text[:-3].strip()
        
        # Direct json.loads
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        
        # Extract first JSON array with regex
        try:
            match = re.search(r"$$.*$$", text, re.DOTALL)
            if match:
                arr_text = match.group(0)
                data = json.loads(arr_text)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        
        print("  ⚠️  Could not parse JSON, returning empty metadata")
        return [{}] * expected_items
    
    # ------------------------------------------------------------------
    # INI: _generate_chunk_metadata
    def _generate_chunk_metadata(self, chunks_batch: List[str], base_metadata: Dict) -> List[Dict]:
        """
        Generate rich metadata for chunks using LLM
        
        Extracts:
        - key_topics
        - entities
        - contains_rule
        - relevance_keywords
        """
        if not self.llm or not chunks_batch:
            return [{}] * len(chunks_batch)
        
        # Build prompt
        chunks_text = "\n\n---CHUNK SEPARATOR---\n\n".join(
            [f"CHUNK {i+1}:\n{chunk[:500]}" for i, chunk in enumerate(chunks_batch)]
        )
        
        prompt = f"""Analyze the following text chunks from a BPJS healthcare regulation document and extract structured metadata for each chunk.

Document: {base_metadata.get('title', 'Unknown')}
Type: {base_metadata.get('document_type', 'Unknown')}

{chunks_text}

For EACH chunk, provide a JSON object with:
1. "chunk_id": index of the chunk (0-based)
2. "key_topics": List of 2-3 main topics (e.g., ["fraud_penalties", "verification_procedure"])
3. "entities": List of mentioned entities like ICD codes, article numbers, fraud types, or tariff codes
4. "contains_rule": true/false - does it contain actionable rules or thresholds?
5. "relevance_keywords": List of 3-5 keywords for semantic search

Respond with a JSON array (one object per chunk), like:
[
  {{
    "chunk_id": 0,
    "key_topics": ["topic1", "topic2"],
    "entities": ["entity1", "entity2"],
    "contains_rule": true,
    "relevance_keywords": ["keyword1", "keyword2", "keyword3"]
  }},
  ...
]

IMPORTANT:
- Return ONLY valid JSON, no explanation text.
- Do NOT wrap the JSON in markdown fences.
"""
        try:
            response = self.llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            
            # Debug (opsional)
            # print("  LLM raw response:", raw_text[:200])
            
            metadata_list = self._parse_json_safely(raw_text, expected_items=len(chunks_batch))
            
            # Pad kalau kurang
            if len(metadata_list) < len(chunks_batch):
                while len(metadata_list) < len(chunks_batch):
                    metadata_list.append({})
            elif len(metadata_list) > len(chunks_batch):
                metadata_list = metadata_list[:len(chunks_batch)]
            
            return metadata_list
        
        except Exception as e:
            print(f"  ⚠️  LLM metadata generation failed: {e}")
            return [{}] * len(chunks_batch)
    
    # ------------------------------------------------------------------
    def chunk_document(self, text: str, metadata: Dict, use_llm: bool = True) -> List[Document]:
        """Chunk a single document with rich metadata"""
        document_type = metadata.get('document_type', 'regulation')
        splitter = self._get_text_splitter(document_type)
        
        chunks = splitter.split_text(text)
        
        # LLM metadata
        if use_llm and self.llm and self.config.ENABLE_LLM_METADATA:
            print(f"  🤖 Generating metadata for {len(chunks)} chunks...")
            llm_metadata_list: List[Dict] = []
            batch_size = self.config.METADATA_BATCH_SIZE
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_meta = self._generate_chunk_metadata(batch, metadata)
                llm_metadata_list.extend(batch_meta)
                time.sleep(0.5)  # rate limit
        else:
            llm_metadata_list = [{}] * len(chunks)
        
        documents: List[Document] = []
        for i, (chunk, llm_meta) in enumerate(zip(chunks, llm_metadata_list)):
            doc_meta = metadata.copy()
            doc_meta.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_token_estimate": len(chunk) // 4,
                "key_topics": llm_meta.get("key_topics", []),
                "entities": llm_meta.get("entities", []),
                "contains_rule": llm_meta.get("contains_rule", False),
                "relevance_keywords": llm_meta.get("relevance_keywords", []),
            })
            documents.append(Document(page_content=chunk, metadata=doc_meta))
        
        return documents
    
    # ------------------------------------------------------------------
    def chunk_all_documents(
        self,
        documents: List[Tuple[str, Dict]],
        use_llm: bool = True
    ) -> List[Document]:
        """Chunk all documents with tqdm progress bar"""
        print("\n" + "="*80)
        print("CHUNKING DOCUMENTS")
        print("="*80)
        
        all_chunks: List[Document] = []
        
        for text, metadata in tqdm(documents, desc="Processing documents", unit="doc"):
            title = metadata.get("title", "Unknown")
            print(f"\n📄 [{title}]")
            print(f"  Document type: {metadata.get('document_type', 'unknown')}")
            print(f"  Total characters: {len(text):,}")
            
            chunks = self.chunk_document(text, metadata, use_llm=use_llm)
            all_chunks.extend(chunks)
            
            print(f"  ✓ Created {len(chunks)} chunks")
        
        print("\n" + "="*80)
        print(f"✅ Total chunks created: {len(all_chunks)}")
        print("="*80)
        
        return all_chunks
