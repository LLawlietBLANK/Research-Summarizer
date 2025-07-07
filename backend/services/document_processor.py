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

    # def _chunk_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """Split text into semantically meaningful chunks with metadata"""
    #     chunks = []
        
    #     for page in pages:
    #         page_text = page["text"]
    #         page_chunks = self.text_splitter.split_text(page_text)
            
    #         for idx, chunk in enumerate(page_chunks):
    #             chunk = re.sub(r'\s+', ' ', chunk).strip()
        


    #             if chunk:
    #                 chunks.append({
    #                     "text": chunk,
    #                     "page_number": page["page_number"],
    #                     "chunk_id": f"p{page['page_number']}c{idx}",
    #                     "metadata": {
    #                         **page["metadata"],
    #                         "char_length": len(chunk),
    #                         "word_count": len(chunk.split())
    #                     }
    #                 })
        
    #     return chunks

    def _chunk_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split text into semantically meaningful chunks with metadata

        Args:
            pages: List of page dictionaries containing text and metadata

        Returns:
            List of processed chunks with:
            - Cleaned text content
            - Page number reference
            - Unique chunk ID
            - Detailed metadata including:
                * Original page metadata
                * Character and word counts
                * Sentence count
                * Has bullet points flag
                * Has numbers flag
        """
        chunks = []

        for page in pages:
            # Pre-process page text
            page_text = self._clean_text(page["text"])

            # Skip empty pages
            if not page_text.strip():
                continue

            # Split into chunks using the text splitter
            page_chunks = self.text_splitter.split_text(page_text)

            for idx, chunk in enumerate(page_chunks):
                # Clean and validate chunk
                clean_chunk = self._clean_chunk(chunk)
                if not clean_chunk:
                    continue

                # Analyze chunk content
                word_count = len(clean_chunk.split())
                char_count = len(clean_chunk)
                sentence_count = len(re.findall(r'[.!?]+[\s\n]|$', clean_chunk))
                has_bullets = bool(re.search(r'^[\s]*[\-•‣⁃]', clean_chunk, re.MULTILINE))
                has_numbers = bool(re.search(r'\b\d+\b', clean_chunk))

                chunks.append({
                    "text": clean_chunk,
                    "page_number": page["page_number"],
                    "chunk_id": f"p{page['page_number']}_c{idx + 1}",  # 1-based index
                    "metadata": {
                        **page.get("metadata", {}),
                        "char_length": char_count,
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "has_bullets": has_bullets,
                        "has_numbers": has_numbers,
                        "avg_word_length": char_count / word_count if word_count > 0 else 0
                    }
                })

        return chunks

    def _clean_text(self, text: str) -> str:
 
        # Normalize whitespace and clean special characters
        text = re.sub(r'\s+', ' ', text)  # Replace all whitespace with single space
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control chars
        text = re.sub(r'-\n', '', text)  # Remove hyphenated line breaks
        text = text.strip()
        return text

    def _clean_chunk(self, chunk: str) -> str:
        """Clean and validate individual chunks"""
        chunk = chunk.strip()
    
        # Skip chunks that are too short or just punctuation
        if len(chunk) < 25 or re.fullmatch(r'[\s\W]+', chunk):
            return ""
        
        # Ensure proper sentence boundaries
        if not chunk[0].isupper():
            chunk = chunk[0].upper() + chunk[1:]
        
        if not chunk.endswith(('.', '!', '?')):
            # Find last sentence boundary
            last_punct = max(
                chunk.rfind('.'),
                chunk.rfind('!'),
                chunk.rfind('?')
            )
            if last_punct > 0:
                chunk = chunk[:last_punct + 1]
            
        return chunk