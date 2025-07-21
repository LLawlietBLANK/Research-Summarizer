from transformers import pipeline
import random
from typing import List, Dict, Any
class ChallengeService:
    def __init__(self):
        self.generator = pipeline(
            "text2text-generation",
            model="mrm8488/t5-base-finetuned-question-generation-ap"
        )
    
    def generate_questions(self, chunks: List[Dict[str, Any]], num_questions: int = 3) -> List[dict]:
        """Generate comprehension questions from document chunks"""
        # Select random chunks to generate questions from
        selected_chunks = random.sample(chunks, min(num_questions, len(chunks)))
        questions = []
        

        for chunk in selected_chunks:
            # Generate question from chunk text
            result = self.generator(
                f"generate question: {chunk['text']}",
                max_length=128,
                num_return_sequences=1
            )
            
            questions.append({
                "question": result[0]['generated_text'],
                "context": chunk['text'],
                "page_number": chunk['page_number'],
                "chunk_id": chunk['chunk_id']
            })
        
        return questions        