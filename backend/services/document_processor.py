import fitz as pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize embeddings for vector search
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with metadata"""
        try:
            doc = pymupdf.open(file_path)
            text_chunks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    text_chunks.append({
                        'text': text,
                        'page': page_num + 1,
                        'source': 'pdf'
                    })
            #testing
            print(text_chunks)


            doc.close()
            logger.info(f"Extracted text from {len(text_chunks)} pages")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return [{
                'text': text,
                'page': 1,
                'source': 'txt'
            }]
            
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {str(e)}")
            raise
    
    def chunk_text(self, text_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = []
        
        for doc_chunk in text_data:
            text = doc_chunk['text']
            page = doc_chunk['page']
            source = doc_chunk['source']
            
            # Split text into smaller chunks
            text_chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(text_chunks):
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append({
                        'text': chunk,
                        'page': page,
                        'source': source,
                        'chunk_id': f"{page}_{i+1}",
                        'metadata': {
                            'page': page,
                            'source': source,
                            'chunk_index': i
                        }
                    })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> FAISS:
        """Create FAISS vector store from chunks"""
        try:
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            logger.info("Created vector store successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def process_document(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Main method to process document"""
        try:
            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                text_data = self.extract_text_from_pdf(file_path)
            elif filename.lower().endswith('.txt'):
                text_data = self.extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Chunk the text
            chunks = self.chunk_text(text_data)
            
            if not chunks:
                raise ValueError("No text content found in document")
            
            # Create vector store for semantic search
            vector_store = self.create_vector_store(chunks)
            
            # Add vector store to each chunk for later retrieval
            for chunk in chunks:
                chunk['vector_store'] = vector_store
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    def semantic_search(self, query: str, chunks: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic search on document chunks"""
        try:
            if not chunks:
                return []
            
            # Use vector store from first chunk (they all share the same one)
            vector_store = chunks[0]['vector_store']
            
            # Perform similarity search
            results = vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            relevant_chunks = []
            for doc, score in results:
                # Find the original chunk
                for chunk in chunks:
                    if chunk['text'] == doc.page_content:
                        relevant_chunks.append({
                            **chunk,
                            'similarity_score': score
                        })
                        break
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return chunks[:k]  # Fallback to first k chunks