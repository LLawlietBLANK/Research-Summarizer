import torch
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class SummarizationService:
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="philschmid/bart-large-cnn-samsum", #facebook bart cnn model was not generating good summaries
            device=0 if torch.cuda.is_available() else -1
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        self.max_combined_length = 4000  # For final summary compression

    def summarize(self, text: str) -> str:
        """Generate comprehensive 100-150 word summary using strategic chunk processing"""
        # Clean and validate input
        clean_text = self._preprocess_text(text)
        if not clean_text.strip():
            return "No meaningful content to summarize"

        # Split document while preserving structure
        chunks = self.text_splitter.split_text(clean_text)
        
        # Process all chunks with importance weighting
        chunk_summaries = []
        for chunk in chunks:
            summary = self._summarize_chunk(chunk)
            if summary:
                chunk_summaries.append(summary)
        
        # Combine and refine
        return self._create_final_summary(chunk_summaries)

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text cleaning with structure preservation"""
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines
        text = re.sub(r'-\n', '', text)  # Join hyphenated words
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def _summarize_chunk(self, chunk: str) -> str:
        """Generate chunk summary with quality controls"""
        chunk = chunk.strip()
        if len(chunk.split()) < 25:  # Skip small fragments
            return ""
            
        return self.summarizer(
            chunk,
            max_length=75,
            min_length=30,
            do_sample=False,
            truncation=True
        )[0]['summary_text']

    def _create_final_summary(self, chunk_summaries: list) -> str:
        """Produce final 100-150 word summary from chunk summaries"""
        combined = " ".join(chunk_summaries)
        
        # First compression pass
        stage1 = self.summarizer(
            combined[:self.max_combined_length],  # Stay within model limits
            max_length=150,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
        
        # Precision adjustment
        return self._adjust_length(stage1)

    def _adjust_length(self, summary: str) -> str:
        """Ensure exact 100-150 word output"""
        words = summary.split()
        if 100 <= len(words) <= 150:
            return summary
            
        # Intelligent truncation prioritizing early content
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        result = []
        word_count = 0
        
        for sent in sentences:
            sent_words = sent.split()
            if word_count + len(sent_words) <= 150:
                result.append(sent)
                word_count += len(sent_words)
                if word_count >= 100:
                    break
        
        return ' '.join(result) or summary[:500]  # Fallback
    


#This is an altervative approach for Writing the summarization service - summarize method (It was generating longer and better summary)

    # def summarize(self, text: str, max_length: int = 300) -> str:
    #     """Generate summary with length control and chunk handling"""
    #     try:
    #         # Handle long documents with chunking
    #         if len(text) > 2000:
    #             chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    #             summaries = [self.model(chunk, max_length=max_length//len(chunks))[0]["summary_text"] for chunk in chunks]
    #             return " ".join(summaries)
    #         return self.model(text, max_length=max_length)[0]["summary_text"]
    #     except Exception as e:
    #         logger.error(f"Summarization error: {str(e)}")
    #         return text[:300] + "..." if len(text) > 300 else text
