# """
# Document Summarization Service
# Combines text extraction, processing, and summarization capabilities.
# """

# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# from typing import List, Dict, Any
# import logging
# import fitz as pymupdf

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# # Configure logging
# logger = logging.getLogger(__name__)

# class SummarizationService:
#     """Handles text summarization using transformer models"""
    
#     def __init__(self, model_name: str = "google/long-t5-tglobal-base"):
#         """
#         Initialize summarization service with specified model
#         Args:
#             model_name: HuggingFace model identifier
#         """
#         self.model_name = model_name
#         self.max_length = 512
#         self.min_length = 50
#         self.device = 0 if torch.cuda.is_available() else -1
        
#         try:
#             logger.info(f"Initializing summarization model: {self.model_name}")
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
#             self.summarizer = pipeline(
#                 "summarization",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=self.device,
#                 framework="pt"
#             )
#             logger.info(f"Model loaded on {'GPU' if self.device == 0 else 'CPU'}")

#         except Exception as e:
#             logger.error(f"Failed to load {self.model_name}: {str(e)}")
#             logger.info("Falling back to facebook/bart-large-cnn")
#             self.summarizer = pipeline(
#                 "summarization",
#                 model="facebook/bart-large-cnn",
#                 device=self.device
#             )

#     def preprocess_text(self, text: str) -> str:
#         """Clean and prepare text for summarization"""
#         text = ' '.join(text.split())
#         tokens = self.tokenizer.encode(text, truncation=True, max_length=4000)
#         return self.tokenizer.decode(tokens, skip_special_tokens=True)

#     def chunk_text_for_summarization(self, text: str, max_chunk_size: int = 3000) -> List[str]:
#         """Split long text into manageable chunks"""
#         words = text.split()
#         chunks, current_chunk = [], []
#         current_length = 0
        
#         for word in words:
#             current_chunk.append(word)
#             current_length += len(word) + 1
#             if current_length >= max_chunk_size:
#                 chunks.append(' '.join(current_chunk))
#                 current_chunk, current_length = [], 0
        
#         if current_chunk:
#             chunks.append(' '.join(current_chunk))
        
#         return chunks

#     def summarize_chunk(self, text: str) -> str:
#         """Generate summary for a single text chunk"""
#         try:
#             if not text.strip():
#                 return ""
            
#             text = self.preprocess_text(text)
#             if len(text) < 50:
#                 return text
            
#             summary = self.summarizer(
#                 text,
#                 max_length=min(self.max_length, len(text.split()) // 2),
#                 min_length=self.min_length,
#                 do_sample=False,
#                 truncation=True
#             )
#             return summary[0]['summary_text']
            
#         except Exception as e:
#             logger.error(f"Chunk summarization failed: {str(e)}")
#             return text[:200] + "..." if len(text) > 200 else text

#     def generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
#         """Generate comprehensive summary from document chunks"""
#         try:
#             full_text = ' '.join(chunk['text'] for chunk in chunks)
#             logger.info(f"Processing text with {len(full_text)} characters")
            
#             if len(full_text.split()) <= 1000:
#                 return self.summarize_chunk(full_text)
            
#             # Multi-stage summarization for long documents
#             text_chunks = self.chunk_text_for_summarization(full_text)
#             chunk_summaries = [
#                 self.summarize_chunk(chunk) 
#                 for chunk in text_chunks
#                 if self.summarize_chunk(chunk).strip()
#             ]
            
#             if not chunk_summaries:
#                 return "No summary generated."
                
#             combined_summaries = ' '.join(chunk_summaries)
#             final_summary = (
#                 self.summarize_chunk(combined_summaries)
#                 if len(combined_summaries.split()) > 500 
#                 else combined_summaries
#             )
            
#             # Trim to reasonable length
#             words = final_summary.split()
#             return ' '.join(words[:180]) + ("..." if len(words) > 180 else "")
            
#         except Exception as e:
#             logger.error(f"Summary generation failed: {str(e)}")
#             sentences = full_text.split('.')[:3]
#             return '. '.join(sentences) + ("." if sentences else "Summary unavailable")

# class DocumentProcessor:
#     """Handles document text extraction and processing"""
    
#     def __init__(self):
#         """Initialize with default text splitter and embeddings"""
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )

#     def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
#         """Extract text and metadata from PDF file"""
#         try:
#             with pymupdf.open(file_path) as doc:
#                 return [
#                     {
#                         'text': page.get_text().strip(),
#                         'page': page_num + 1,
#                         'source': 'pdf'
#                     }
#                     for page_num, page in enumerate(doc)
#                     if page.get_text().strip()
#                 ]
#         except Exception as e:
#             logger.error(f"PDF extraction failed: {str(e)}")
#             raise

#     def extract_text_from_txt(self, file_path: str) -> List[Dict[str, Any]]:
#         """Extract text from plain text file"""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 return [{
#                     'text': file.read(),
#                     'page': 1,
#                     'source': 'txt'
#                 }]
#         except Exception as e:
#             logger.error(f"TXT extraction failed: {str(e)}")
#             raise

#     def chunk_text(self, text_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Split document text into manageable chunks with metadata"""
#         chunks = []
#         for doc_chunk in text_data:
#             chunks.extend(
#                 {
#                     'text': chunk,
#                     'page': doc_chunk['page'],
#                     'source': doc_chunk['source'],
#                     'chunk_id': f"{doc_chunk['page']}_{i+1}",
#                     'metadata': {
#                         'page': doc_chunk['page'],
#                         'source': doc_chunk['source'],
#                         'chunk_index': i
#                     }
#                 }
#                 for i, chunk in enumerate(self.text_splitter.split_text(doc_chunk['text']))
#                 if chunk.strip()
#             )
#         logger.info(f"Created {len(chunks)} text chunks")
#         return chunks

#     def create_vector_store(self, chunks: List[Dict[str, Any]]) -> FAISS:
#         """Create searchable vector store from text chunks"""
#         try:
#             vector_store = FAISS.from_texts(
#                 texts=[chunk['text'] for chunk in chunks],
#                 embedding=self.embeddings,
#                 metadatas=[chunk['metadata'] for chunk in chunks]
#             )
#             logger.info("Vector store created successfully")
#             return vector_store
#         except Exception as e:
#             logger.error(f"Vector store creation failed: {str(e)}")
#             raise

#     def process_document(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
#         """Main document processing pipeline"""
#         try:
#             # File type detection and extraction
#             if filename.lower().endswith('.pdf'):
#                 text_data = self.extract_text_from_pdf(file_path)
#             elif filename.lower().endswith('.txt'):
#                 text_data = self.extract_text_from_txt(file_path)
#             else:
#                 raise ValueError(f"Unsupported file type: {filename}")
            
#             # Chunking and vector store creation
#             chunks = self.chunk_text(text_data)
#             if not chunks:
#                 raise ValueError("No processable content found")
            
#             vector_store = self.create_vector_store(chunks)
#             for chunk in chunks:
#                 chunk['vector_store'] = vector_store
            
#             return chunks
            
#         except Exception as e:
#             logger.error(f"Document processing failed: {str(e)}")
#             raise

#     def semantic_search(self, query: str, chunks: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
#         """Find most relevant chunks for a search query"""
#         try:
#             if not chunks or 'vector_store' not in chunks[0]:
#                 return chunks[:k]  # Fallback
                
#             results = chunks[0]['vector_store'].similarity_search_with_score(query, k=k)
#             return [
#                 {
#                     **next((c for c in chunks if c['text'] == doc.page_content), {}),
#                     'similarity_score': score
#                 }
#                 for doc, score in results
#             ]
#         except Exception as e:
#             logger.error(f"Semantic search failed: {str(e)}")
#             return chunks[:k]