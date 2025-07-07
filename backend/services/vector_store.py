import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
    
    def create(self, chunks: List[Dict[str, Any]]):
        """Create FAISS index from document chunks"""
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
    
    def semantic_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant chunks for a query"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        return [self.chunks[i] for i in indices[0]] if indices.size > 0 else []