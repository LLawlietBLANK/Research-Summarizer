# import fitz as pymupdf
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# import os
# from typing import List, Dict, Any
# import logging

# logger = logging.getLogger(__name__)
# class DocumentProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
        
#         # Initialize embeddings for vector search
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )
    
#     def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
#         """Extract text from PDF with metadata"""
#         try:
#             doc = pymupdf.open(file_path)
#             text_chunks = []
            
#             for page_num in range(len(doc)):
#                 page = doc[page_num]
#                 text = page.get_text()
                
#                 if text.strip():  # Only add non-empty pages
#                     text_chunks.append({
#                         'text': text,
#                         'page': page_num + 1,
#                         'source': 'pdf'
#                     })
#             #testing
#             print(text_chunks)


#             doc.close()
#             logger.info(f"Extracted text from {len(text_chunks)} pages")
#             return text_chunks
            
#         except Exception as e:
#             logger.error(f"Error extracting text from PDF: {str(e)}")
#             raise
    
#     def extract_text_from_txt(self, file_path: str) -> List[Dict[str, Any]]:
#         """Extract text from TXT file"""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 text = file.read()
            
#             return [{
#                 'text': text,
#                 'page': 1,
#                 'source': 'txt'
#             }]
            
#         except Exception as e:
#             logger.error(f"Error extracting text from TXT: {str(e)}")
#             raise
    
#     def chunk_text(self, text_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Split text into chunks with metadata"""
#         chunks = []
        
#         for doc_chunk in text_data:
#             text = doc_chunk['text']
#             page = doc_chunk['page']
#             source = doc_chunk['source']
            
#             # Split text into smaller chunks
#             text_chunks = self.text_splitter.split_text(text)
            
#             for i, chunk in enumerate(text_chunks):
#                 if chunk.strip():  # Only add non-empty chunks
#                     chunks.append({
#                         'text': chunk,
#                         'page': page,
#                         'source': source,
#                         'chunk_id': f"{page}_{i+1}",
#                         'metadata': {
#                             'page': page,
#                             'source': source,
#                             'chunk_index': i
#                         }
#                     })
        
#         logger.info(f"Created {len(chunks)} chunks")
#         return chunks
    
#     def create_vector_store(self, chunks: List[Dict[str, Any]]) -> FAISS:
#         """Create FAISS vector store from chunks"""
#         try:
#             texts = [chunk['text'] for chunk in chunks]
#             metadatas = [chunk['metadata'] for chunk in chunks]
            
#             vector_store = FAISS.from_texts(
#                 texts=texts,
#                 embedding=self.embeddings,
#                 metadatas=metadatas
#             )
            
#             logger.info("Created vector store successfully")
#             return vector_store
            
#         except Exception as e:
#             logger.error(f"Error creating vector store: {str(e)}")
#             raise
    
#     def process_document(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
#         """Main method to process document"""
#         try:
#             # Extract text based on file type
#             if filename.lower().endswith('.pdf'):
#                 text_data = self.extract_text_from_pdf(file_path)
#             elif filename.lower().endswith('.txt'):
#                 text_data = self.extract_text_from_txt(file_path)
#             else:
#                 raise ValueError(f"Unsupported file type: {filename}")
            
#             # Chunk the text
#             chunks = self.chunk_text(text_data)
            
#             if not chunks:
#                 raise ValueError("No text content found in document")
            
#             # Create vector store for semantic search
#             vector_store = self.create_vector_store(chunks)
            
#             # Add vector store to each chunk for later retrieval
#             for chunk in chunks:
#                 chunk['vector_store'] = vector_store
            
#             return chunks
            
#         except Exception as e:
#             logger.error(f"Error processing document: {str(e)}")
#             raise
    
#     def semantic_search(self, query: str, chunks: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
#         """Perform semantic search on document chunks"""
#         try:
#             if not chunks:
#                 return []
            
#             # Use vector store from first chunk (they all share the same one)
#             vector_store = chunks[0]['vector_store']
            
#             # Perform similarity search
#             results = vector_store.similarity_search_with_score(query, k=k)
            
#             # Format results
#             relevant_chunks = []
#             for doc, score in results:
#                 # Find the original chunk
#                 for chunk in chunks:
#                     if chunk['text'] == doc.page_content:
#                         relevant_chunks.append({
#                             **chunk,
#                             'similarity_score': score
#                         })
#                         break
            
#             return relevant_chunks
            
#         except Exception as e:
#             logger.error(f"Error in semantic search: {str(e)}")
#             return chunks[:k]  # Fallback to first k chunks
























# import os
# import re
# import fitz  # PyMuPDF
# import numpy as np
# from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import faiss

# class DocumentProcessor:
#     def __init__(self):
#         # Text extraction
#         self.pdf_parser = fitz.open
        
#         # Text processing
#         self.tokenizer = pipeline("summarization", model="facebook/bart-large-cnn")
#         self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Text splitting configuration
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=50,
#             length_function=len,
#             separators=["\n\n", "\n", "(?<=\. )", " ", ""]
#         )
    
#     def extract_text(self, file_path: str) -> tuple:
#         """Extract text from PDF/TXT with structure preservation"""
#         text = ""
#         structure = []
        
#         if file_path.endswith('.pdf'):
#             with self.pdf_parser(file_path) as doc:
#                 for page_num, page in enumerate(doc):
#                     page_text = page.get_text("text", sort=True)
#                     text += page_text + "\n"
                    
#                     # Preserve structure: (page, paragraph)
#                     paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
#                     structure.extend([
#                         (page_num, para_idx, para) 
#                         for para_idx, para in enumerate(paragraphs)
#                     ])
        
#         elif file_path.endswith('.txt'):
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
#             structure = [(0, idx, para) for idx, para in enumerate(paragraphs)]
        
#         return text, structure
    
#     def chunk_text(self, text: str) -> list:
#         """Split text into semantically meaningful chunks"""
#         chunks = self.text_splitter.split_text(text)
        
#         # Further processing to ensure chunk quality
#         cleaned_chunks = []
#         for chunk in chunks:
#             # Remove excessive whitespace
#             chunk = re.sub(r'\s+', ' ', chunk).strip()
#             # Ensure chunk ends with complete sentence
#             if not re.search(r'[.!?]$', chunk):
#                 if '.' in chunk:
#                     chunk = chunk.rsplit('.', 1)[0] + '.'
#             cleaned_chunks.append(chunk)
        
#         return cleaned_chunks
    
#     def generate_embeddings(self, chunks: list) -> faiss.Index:
#         """Create FAISS index for document chunks"""
#         embeddings = self.embedder.encode(chunks)
#         index = faiss.IndexFlatL2(embeddings.shape[1])
#         index.add(np.array(embeddings))
#         return index
    
#     def summarize_document(self, text: str, max_length=150) -> str:
#         """Generate concise summary using T5 model"""
#         # Preprocess text for T5
#         inputs = self.tokenizer(
#             "summarize: " + text,
#             return_tensors="pt",
#             max_length=1024,
#             truncation=True
#         )
        
#         # Generate summary
#         summary_ids = self.summarizer.generate(
#             inputs["input_ids"],
#             max_length=max_length,
#             min_length=30,
#             length_penalty=2.0,
#             num_beams=4,
#             early_stopping=True
#         )
        
#         summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
#         # Ensure word limit
#         words = summary.split()
#         return " ".join(words[:min(len(words), 150)])
    
#     def process_document(self, file_path: str) -> dict:
#         """Full document processing pipeline"""
#         # Validate file
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
        
#         # Process document
#         raw_text, structure = self.extract_text(file_path)
#         chunks = self.chunk_text(raw_text)
#         index = self.generate_embeddings(chunks)
#         summary = self.summarize_document(raw_text)
        
#         return {
#             "raw_text": raw_text,
#             "structure": structure,
#             "chunks": chunks,
#             "index": index,
#             "summary": summary
#         }







import re
import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize with configurable chunking parameters"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

    def process(self, file_bytes: bytes, file_type: str) -> List[Dict[str, Any]]:
        """
        Complete document processing pipeline
        Returns: List of processed chunks with metadata
        """
        # Extract text from the document
        pages = self._extract_text(file_bytes, file_type)
        
        # Split into chunks with metadata
        chunks = self._chunk_text(pages)
        
        return chunks

    def _extract_text(self, file_bytes: bytes, file_type: str) -> List[Dict[str, Any]]:
        """Extract structured text from PDF/TXT bytes"""
        pages = []
        
        if file_type.lower() == 'pdf':
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text", sort=True)
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "metadata": {
                            "width": page.rect.width,
                            "height": page.rect.height
                        }
                    })
        
        elif file_type.lower() == 'txt':
            text = file_bytes.decode('utf-8')
            pages.append({
                "page_number": 1,
                "text": text,
                "metadata": {}
            })
        
        return pages

    def _chunk_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split text into semantically meaningful chunks with metadata"""
        chunks = []
        
        for page in pages:
            page_text = page["text"]
            page_chunks = self.text_splitter.split_text(page_text)
            
            for idx, chunk in enumerate(page_chunks):
                chunk = re.sub(r'\s+', ' ', chunk).strip()
                if chunk:
                    chunks.append({
                        "text": chunk,
                        "page_number": page["page_number"],
                        "chunk_id": f"p{page['page_number']}c{idx}",
                        "metadata": {
                            **page["metadata"],
                            "char_length": len(chunk),
                            "word_count": len(chunk.split())
                        }
                    })
        
        return chunks