from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

class QAService:
    def __init__(self):
        self.model_name = "deepset/roberta-base-squad2"
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Load QA pipeline
            logger.info(f"Loading QA model: {self.model_name}")
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                device=self.device,
                return_all_scores=True
            )
            
            # Load tokenizer for text processing
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("QA service initialized successfully")
            
        except Exception as e:
            logger.warning(f"Error loading {self.model_name}, falling back to default model: {str(e)}")
            # Fallback to a more widely available model
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=self.device
            )
    
    def preprocess_question(self, question: str) -> str:
        """Preprocess question for better QA performance"""
        # Remove extra whitespace
        question = ' '.join(question.split())
        
        # Ensure question ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        return question
    
    def find_relevant_context(self, question: str, chunks: List[Dict[str, Any]], max_chunks: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant chunks for the question using semantic search"""
        try:
            # Use semantic search if available
            if chunks and 'vector_store' in chunks[0]:
                from services.document_processor import DocumentProcessor
                doc_processor = DocumentProcessor()
                relevant_chunks = doc_processor.semantic_search(question, chunks, k=max_chunks)
                return relevant_chunks
            
            # Fallback: simple keyword matching
            question_words = set(question.lower().split())
            chunk_scores = []
            
            for chunk in chunks:
                chunk_words = set(chunk['text'].lower().split())
                # Calculate simple overlap score
                overlap = len(question_words.intersection(chunk_words))
                chunk_scores.append((chunk, overlap))
            
            # Sort by score and return top chunks
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in chunk_scores[:max_chunks]]
            
        except Exception as e:
            logger.error(f"Error finding relevant context: {str(e)}")
            return chunks[:max_chunks]
    
    def answer_from_context(self, question: str, context: str) -> Dict[str, Any]:
        """Answer question from a specific context"""
        try:
            # Preprocess inputs
            question = self.preprocess_question(question)
            
            # Truncate context if too long
            max_context_length = 2000
            if len(context) > max_context_length:
                context = context[:max_context_length]
            
            # Get answer from QA pipeline
            result = self.qa_pipeline(
                question=question,
                context=context
            )
            
            # Extract answer and confidence
            if isinstance(result, list):
                best_result = result[0]
            else:
                best_result = result
            
            answer = best_result['answer']
            confidence = best_result['score']
            
            # Find the context window around the answer
            answer_start = context.find(answer)
            if answer_start != -1:
                # Get surrounding context (50 chars before and after)
                start_idx = max(0, answer_start - 50)
                end_idx = min(len(context), answer_start + len(answer) + 50)
                source_context = context[start_idx:end_idx]
            else:
                source_context = context[:100] + "..."
            
            return {
                'answer': answer,
                'confidence': confidence,
                'source_context': source_context
            }
            
        except Exception as e:
            logger.error(f"Error answering from context: {str(e)}")
            return {
                'answer': "I couldn't find an answer to your question in the provided context.",
                'confidence': 0.0,
                'source_context': ""
            }
    
    def answer_question(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Answer a question using the document chunks"""
        try:
            if not chunks:
                return {
                    'answer': "No document content available to answer your question.",
                    'confidence': 0.0,
                    'source_reference': ""
                }
            
            logger.info(f"Answering question: {question}")
            
            # Find relevant chunks
            relevant_chunks = self.find_relevant_context(question, chunks)
            
            if not relevant_chunks:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'confidence': 0.0,
                    'source_reference': ""
                }
            
            # Try to answer using each relevant chunk
            best_answer = None
            best_confidence = 0.0
            best_source_ref = ""
            
            for chunk in relevant_chunks:
                result = self.answer_from_context(question, chunk['text'])
                
                if result['confidence'] > best_confidence:
                    best_answer = result['answer']
                    best_confidence = result['confidence']
                    best_source_ref = self.create_source_reference(chunk, result['source_context'])
            
            # If no good answer found, try with combined context
            if best_confidence < 0.3:
                combined_context = ' '.join([chunk['text'] for chunk in relevant_chunks[:2]])
                result = self.answer_from_context(question, combined_context)
                
                if result['confidence'] > best_confidence:
                    best_answer = result['answer']
                    best_confidence = result['confidence']
                    best_source_ref = f"Multiple sections (pages {', '.join([str(chunk['page']) for chunk in relevant_chunks[:2]])})"
            
            # Final check for answer quality
            if best_confidence < 0.1:
                best_answer = "I couldn't find a confident answer to your question in the document."
                best_source_ref = "No reliable source found"
            
            return {
                'answer': best_answer,
                'confidence': round(best_confidence, 3),
                'source_reference': best_source_ref
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': "An error occurred while processing your question.",
                'confidence': 0.0,
                'source_reference': ""
            }
    
    def create_source_reference(self, chunk: Dict[str, Any], context: str) -> str:
        """Create a source reference for the answer"""
        try:
            page_info = f"Page {chunk['page']}" if chunk['page'] > 0 else "Document"
            
            # Clean up context for reference
            context = context.strip()
            if len(context) > 100:
                context = context[:100] + "..."
            
            return f"{page_info} - \"{context}\""
            
        except Exception as e:
            logger.error(f"Error creating source reference: {str(e)}")
            return "Source reference unavailable"
    
    def validate_answer(self, answer: str, question: str) -> bool:
        """Validate if the answer is reasonable for the question"""
        try:
            # Basic validation rules
            if not answer or len(answer.strip()) < 2:
                return False
            
            # Check if answer is just a repetition of the question
            if answer.lower() in question.lower():
                return False
            
            # Check for common non-answers
            non_answers = ["i don't know", "no answer", "not found", "unclear"]
            if any(na in answer.lower() for na in non_answers):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            return False