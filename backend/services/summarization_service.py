from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SummarizationService:
    def __init__(self):
        self.model_name = "google/long-t5-tglobal-base"
        self.max_length = 512
        self.min_length = 50
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Load tokenizer and model
            logger.info(f"Loading summarization model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                framework="pt"
            )

            print("CUDA available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("Using GPU:", torch.cuda.get_device_name(0))
            else:
                print("Using CPU")
            
            logger.info("Summarization service initialized successfully")
            
        except Exception as e:
            logger.warning(f"Error loading {self.model_name}, falling back to default model: {str(e)}")
            # Fallback to a more widely available model
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=self.device
            )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization"""
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Ensure text is not too long for the model
        max_tokens = 4000  # Conservative limit
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return text
    
    def chunk_text_for_summarization(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """Split text into chunks for summarization"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_chunk(self, text: str) -> str:
        """Summarize a single chunk of text"""
        try:
            # Ensure text is not empty
            if not text.strip():
                return ""
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            if len(text) < 50:  # Too short to summarize
                return text
            
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=min(self.max_length, len(text.split()) // 2),
                min_length=self.min_length,
                do_sample=False,
                truncation=True
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            # Return truncated version as fallback
            return text[:200] + "..." if len(text) > 200 else text
    
    def generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive summary from document chunks"""
        try:
            # Combine all chunk texts
            full_text = ' '.join([chunk['text'] for chunk in chunks])
            
            logger.info(f"Generating summary for text of length: {len(full_text)}")
            
            # If text is short enough, summarize directly
            if len(full_text.split()) <= 1000:
                return self.summarize_chunk(full_text)
            
            # For longer texts, use a two-step approach
            # Step 1: Summarize each chunk
            chunk_summaries = []
            text_chunks = self.chunk_text_for_summarization(full_text)
            
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Summarizing chunk {i+1}/{len(text_chunks)}")
                summary = self.summarize_chunk(chunk)
                if summary.strip():
                    chunk_summaries.append(summary)
            
            # Step 2: Combine and summarize the chunk summaries
            if len(chunk_summaries) > 1:
                combined_summaries = ' '.join(chunk_summaries)
                
                # If combined summaries are still too long, summarize again
                if len(combined_summaries.split()) > 500:
                    final_summary = self.summarize_chunk(combined_summaries)
                else:
                    final_summary = combined_summaries
            else:
                final_summary = chunk_summaries[0] if chunk_summaries else "No summary available."
            
            # Ensure summary is around 150 words
            words = final_summary.split()
            if len(words) > 180:
                final_summary = ' '.join(words[:180]) + "..."
            
            logger.info(f"Generated summary with {len(final_summary.split())} words")
            return final_summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fallback: return first few sentences
            sentences = full_text.split('.')[:3]
            return '. '.join(sentences) + "." if sentences else "Summary generation failed."
    
    def extract_key_points(self, text: str, num_points: int = 3) -> List[str]:
        """Extract key points from text"""
        try:
            # Split text into sentences
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # For simplicity, return first few sentences as key points
            # In production, you might want to use more sophisticated extraction
            key_points = sentences[:num_points]
            
            return key_points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return []
