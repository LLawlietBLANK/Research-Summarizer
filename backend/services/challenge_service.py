# from transformers import pipeline
# import torch
# from typing import List, Dict, Any
# import logging
# import random
# import re

# logger = logging.getLogger(__name__)

# class ChallengeService:
#     def __init__(self):
#         self.device = 0 if torch.cuda.is_available() else -1
        
#         try:
#             # Load text generation pipeline for question generation
#             logger.info("Loading text generation model for challenge questions")
#             self.question_generator = pipeline(
#                 "text2text-generation",
#                 model="t5-base",
#                 device=self.device
#             )
            
#             # Load QA pipeline for answer validation
#             self.qa_pipeline = pipeline(
#                 "question-answering",
#                 model="deepset/roberta-base-squad2",
#                 device=self.device
#             )
            
#             logger.info("Challenge service initialized successfully")
            
#         except Exception as e:
#             logger.warning(f"Error loading models: {str(e)}")
#             # Fallback to basic question generation
#             self.question_generator = None
#             self.qa_pipeline = None
    
#     def extract_key_sentences(self, chunks: List[Dict[str, Any]], num_sentences: int = 10) -> List[str]:
#         """Extract key sentences from document chunks"""
#         try:
#             all_sentences = []
            
#             for chunk in chunks:
#                 text = chunk['text']
#                 # Split into sentences
#                 sentences = re.split(r'[.!?]+', text)
#                 sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
                
#                 # Add metadata to sentences
#                 for sentence in sentences:
#                     all_sentences.append({
#                         'text': sentence,
#                         'page': chunk['page'],
#                         'chunk_id': chunk['chunk_id']
#                     })
            
#             # Sort by length and informativeness (simple heuristic)
#             all_sentences.sort(key=lambda x: len(x['text']), reverse=True)
            
#             # Return top sentences
#             key_sentences = all_sentences[:num_sentences]
#             return [s['text'] for s in key_sentences]
            
#         except Exception as e:
#             logger.error(f"Error extracting key sentences: {str(e)}")
#             return []
    
#     def generate_comprehension_question(self, sentence: str) -> Dict[str, Any]:
#         """Generate a comprehension question from a sentence"""
#         try:
#             # Question templates based on sentence patterns
#             templates = [
#                 {"pattern": r"(\w+) is (\w+)", "template": "What is {0}?", "answer_pattern": "{1}"},
#                 {"pattern": r"(\w+) was (\w+)", "template": "What was {0}?", "answer_pattern": "{1}"},
#                 {"pattern": r"(\w+) can (\w+)", "template": "What can {0} do?", "answer_pattern": "{1}"},
#                 {"pattern": r"(\w+) will (\w+)", "template": "What will {0} do?", "answer_pattern": "{1}"},
#                 {"pattern": r"The (\w+) of (\w+) is (\w+)", "template": "What is the {0} of {1}?", "answer_pattern": "{2}"},
#                 {"pattern": r"(\w+) occurs when (\w+)", "template": "When does {0} occur?", "answer_pattern": "When {1}"},
#                 {"pattern": r"(\w+) because (\w+)", "template": "Why does {0} happen?", "answer_pattern": "Because {1}"},
#             ]
            
#             # Try to match patterns
#             for template_info in templates:
#                 match = re.search(template_info["pattern"], sentence, re.IGNORECASE)
#                 if match:
#                     groups = match.groups()
#                     question = template_info["template"].format(*groups)
#                     answer = template_info["answer_pattern"].format(*groups)
                    
#                     return {
#                         'question': question,
#                         'answer': answer,
#                         'source_sentence': sentence,
#                         'type': 'comprehension'
#                     }
            
#             # Fallback: generate generic questions
#             if len(sentence.split()) > 10:
#                 # Extract key entity from sentence
#                 words = sentence.split()
#                 key_entity = words[0]  # Simple heuristic
                
#                 return {
#                     'question': f"What does the text say about {key_entity}?",
#                     'answer': sentence,
#                     'source_sentence': sentence,
#                     'type': 'comprehension'
#                 }
            
#             return None
            
#         except Exception as e:
#             logger.error(f"Error generating comprehension question: {str(e)}")
#             return None
    
#     def generate_logical_question(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Generate logical reasoning questions"""
#         try:
#             # Find sentences with logical connectors
#             logical_patterns = [
#                 r"because",
#                 r"therefore",
#                 r"however",
#                 r"although",
#                 r"since",
#                 r"as a result",
#                 r"consequently"
#             ]
            
#             logical_sentences = []
#             for chunk in chunks:
#                 text = chunk['text']
#                 for pattern in logical_patterns:
#                     if re.search(pattern, text, re.IGNORECASE):
#                         sentences = re.split(r'[.!?]+', text)
#                         for sentence in sentences:
#                             if re.search(pattern, sentence, re.IGNORECASE):
#                                 logical_sentences.append({
#                                     'text': sentence.strip(),
#                                     'page': chunk['page']
#                                 })
            
#             if logical_sentences:
#                 selected = random.choice(logical_sentences)
#                 sentence = selected['text']
                
#                 # Generate logical question
#                 if "because" in sentence.lower():
#                     parts = sentence.split("because")
#                     if len(parts) == 2:
#                         return {
#                             'question': f"Why {parts[0].strip().lower()}?",
#                             'answer': f"Because {parts[1].strip()}",
#                             'source_sentence': sentence,
#                             'type': 'logical'
#                         }
                
#                 elif "therefore" in sentence.lower():
#                     return {
#                         'question': f"What can be concluded from the information in this sentence: '{sentence}'?",
#                         'answer': "The conclusion indicated by 'therefore' in the sentence",
#                         'source_sentence': sentence,
#                         'type': 'logical'
#                     }
            
#             return None
            
#         except Exception as e:
#             logger.error(f"Error generating logical question: {str(e)}")
#             return None
    
#     def generate_factual_question(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Generate factual questions"""
#         try:
#             # Look for sentences with specific facts (numbers, dates, names)
#             fact_patterns = [
#                 r"\b\d{4}\b",  # Years
#                 r"\b\d+%\b",   # Percentages
#                 r"\b\d+\b",    # Numbers
#                 r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Proper names
#             ]
            
#             factual_sentences = []
#             for chunk in chunks:
#                 text = chunk['text']
#                 sentences = re.split(r'[.!?]+', text)
                
#                 for sentence in sentences:
#                     sentence = sentence.strip()
#                     if len(sentence) > 20:
#                         for pattern in fact_patterns:
#                             if re.search(pattern, sentence):
#                                 factual_sentences.append({
#                                     'text': sentence,
#                                     'page': chunk['page']
#                                 })
#                                 break
            
#             if factual_sentences:
#                 selected = random.choice(factual_sentences)
#                 sentence = selected['text']
                
#                 # Generate question based on fact type
#                 if re.search(r"\b\d{4}\b", sentence):  # Year
#                     year = re.search(r"\b\d{4}\b", sentence).group()
#                     return {
#                         'question': f"What happened in {year} according to the document?",
#                         'answer': sentence,
#                         'source_sentence': sentence,
#                         'type': 'factual'
#                     }
                
#                 elif re.search(r"\b\d+%\b", sentence):  # Percentage
#                     percentage = re.search(r"\b\d+%\b", sentence).group()
#                     return {
#                         'question': f"What does the {percentage} mentioned in the document refer to?",
#                         'answer': sentence,
#                         'source_sentence': sentence,
#                         'type': 'factual'
#                     }
                
#                 else:
#                     # Generic factual question
#                     return {
#                         'question': f"What specific fact is mentioned in this part of the document?",
#                         'answer': sentence,
#                         'source_sentence': sentence,
#                         'type': 'factual'
#                     }
            
#             return None
            
#         except Exception as e:
#             logger.error(f"Error generating factual question: {str(e)}")
#             return None
    
#     def generate_questions(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Generate 3 challenge questions from document chunks"""
#         try:
#             questions = []
            
#             # Generate different types of questions
#             question_generators = [
#                 self.generate_comprehension_question,
#                 self.generate_logical_question,
#                 self.generate_factual_question
#             ]
            
#             # Extract key sentences for comprehension questions
#             key_sentences = self.extract_key_sentences(chunks, 20)
            
#             # Generate comprehension question
#             if key_sentences:
#                 for sentence in key_sentences:
#                     comp_question = self.generate_comprehension_question(sentence)
#                     if comp_question:
#                         questions.append(comp_question)
#                         break
            
#             # Generate logical question
#             logical_question = self.generate_logical_question(chunks)
#             if logical_question:
#                 questions.append(logical_question)
            
#             # Generate factual question
#             factual_question = self.generate_factual_question(chunks)
#             if factual_question:
#                 questions.append(factual_question)
            
#             # Fill remaining slots with more comprehension questions
#             while len(questions) < 3 and key_sentences:
#                 for sentence in key_sentences:
#                     if len(questions) >= 3:
#                         break
#                     comp_question = self.generate_comprehension_question(sentence)
#                     if comp_question and comp_question not in questions:
#                         questions.append(comp_question)
            
#             # Fallback questions if not enough generated
#             if len(questions) < 3:
#                 fallback_questions = [
#                     {
#                         'question': "What is the main topic discussed in this document?",
#                         'answer': "The main topic as described in the document",
#                         'source_sentence': "",
#                         'type': 'general'
#                     },
#                     {
#                         'question': "What are the key points mentioned in the document?",
#                         'answer': "The key points as outlined in the document",
#                         'source_sentence': "",
#                         'type': 'general'
#                     },
#                     {
#                         'question': "What conclusion can be drawn from the document?",
#                         'answer': "The conclusion as presented in the document",
#                         'source_sentence': "",
#                         'type': 'general'
#                     }
#                 ]
                
#                 for fallback in fallback_questions:
#                     if len(questions) >= 3:
#                         break
#                     questions.append(fallback)
            
#             return questions[:3]
            
#         except Exception as e:
#             logger.error(f"Error generating questions: {str(e)}")
#             # Return default questions as fallback
#             return [
#                 {
#                     'question': "What is the main topic of this document?",
#                     'answer': "The main topic as described in the document",
#                     'source_sentence': "",
#                     'type': 'general'
#                 },
#                 {
#                     'question': "What are the important details mentioned?",
#                     'answer': "The important details as outlined in the document",
#                     'source_sentence': "",
#                     'type': 'general'
#                 },
#                 {
#                     'question': "What can you conclude from reading this document?",
#                     'answer': "The conclusion as presented in the document",
#                     'source_sentence': "",
#                     'type': 'general'
#                 }
#             ]
    
#     def evaluate_answer(self, question_data: Dict[str, Any], user_answer: str, document_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Evaluate user's answer to a challenge question"""
#         try:
#             correct_answer = question_data['answer']
#             question = question_data['question']
            
#             # Clean and normalize answers
#             user_answer = user_answer.strip().lower()
#             correct_answer = correct_answer.strip().lower()
            
#             # Basic similarity scoring
#             score = 0
#             feedback = ""
            
#             # Exact match
#             if user_answer == correct_answer:
#                 score = 10
#                 feedback = "Excellent! Your answer is exactly correct."
            
#             # Partial match - check for key words
#             elif user_answer in correct_answer or correct_answer in user_answer:
#                 score = 7
#                 feedback = "Good! Your answer contains the key information."
            
#             # Check for semantic similarity using key words
#             else:
#                 user_words = set(user_answer.split())
#                 correct_words = set(correct_answer.split())
                
#                 # Remove common words
#                 stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be'}
#                 user_words = user_words - stop_words
#                 correct_words = correct_words - stop_words
                
#                 if user_words and correct_words:
#                     overlap = len(user_words.intersection(correct_words))
#                     similarity = overlap / len(correct_words.union(user_words))
                    
#                     if similarity > 0.5:
#                         score = 5
#                         feedback = "Your answer is partially correct. You got some key points."
#                     elif similarity > 0.2:
#                         score = 3
#                         feedback = "Your answer has some relevant information but misses key points."
#                     else:
#                         score = 1
#                         feedback = "Your answer doesn't match the expected response well."
#                 else:
#                     score = 1
#                     feedback = "Your answer doesn't contain the key information."
            
#             # Try to use QA pipeline for additional validation if available
#             if self.qa_pipeline and question_data.get('source_sentence'):
#                 try:
#                     qa_result = self.qa_pipeline(
#                         question=question,
#                         context=question_data['source_sentence']
#                     )
                    
#                     pipeline_answer = qa_result['answer'].lower()
                    
#                     # Check if user answer matches pipeline answer
#                     if user_answer in pipeline_answer or pipeline_answer in user_answer:
#                         score = max(score, 8)
#                         feedback = "Great! Your answer aligns well with the document content."
                        
#                 except Exception as e:
#                     logger.error(f"Error in QA validation: {str(e)}")
            
#             return {
#                 'score': score,
#                 'feedback': feedback,
#                 'correct_answer': question_data['answer']
#             }
            
#         except Exception as e:
#             logger.error(f"Error evaluating answer: {str(e)}")
#             return {
#                 'score': 0,
#                 'feedback': "Error occurred while evaluating your answer.",
#                 'correct_answer': question_data.get('answer', 'Answer not available')
#             }





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
                max_length=64,
                num_return_sequences=1
            )
            
            questions.append({
                "question": result[0]['generated_text'],
                "context": chunk['text'],
                "page_number": chunk['page_number'],
                "chunk_id": chunk['chunk_id']
            })
        
        return questions