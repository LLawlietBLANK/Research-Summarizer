
from transformers import pipeline

class QAService:
    def __init__(self):
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
    
    def answer(self, question: str, context: str) -> dict:
        """Answer question based on context"""
        result = self.qa_pipeline(question=question, context=context)
        return {
            "answer": result["answer"],
            "score": result["score"]
        }