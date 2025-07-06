# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn
# import os
# import tempfile
# from typing import List, Dict, Any
# import logging

# from services.document_processor import DocumentProcessor
# from services.summarization_service import SummarizationService
# from services.qa_service import QAService
# from services.challenge_service import ChallengeService

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Smart Research Assistant", version="1.0.0")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize services
# doc_processor = DocumentProcessor()
# summarization_service = SummarizationService()
# qa_service = QAService()
# challenge_service = ChallengeService()

# # Store for uploaded documents (in production, use a database)
# document_store = {}

# class QuestionRequest(BaseModel):
#     document_id: str
#     question: str

# class ChallengeAnswerRequest(BaseModel):
#     document_id: str
#     question_id: int
#     answer: str

# class DocumentResponse(BaseModel):
#     document_id: str
#     filename: str
#     summary: str
#     chunk_count: int

# class AnswerResponse(BaseModel):
#     answer: str
#     confidence: float
#     source_reference: str

# class ChallengeQuestion(BaseModel):
#     id: int
#     question: str

# class ChallengeResponse(BaseModel):
#     questions: List[ChallengeQuestion]

# class EvaluationResponse(BaseModel):
#     score: int
#     feedback: str
#     correct_answer: str






# @app.post("/upload", response_model=DocumentResponse)
# async def upload_document(file: UploadFile = File(...)):
#     """Upload and process a document (PDF or TXT)"""
#     try:
#         # Validate file type
#         if not file.filename.endswith(('.pdf', '.txt')):
#             raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
#             content = await file.read()
#             tmp_file.write(content)
#             tmp_file_path = tmp_file.name
        
#         try:
#             # Process document
#             logger.info(f"Processing document: {file.filename}")
#             chunks = doc_processor.process_document(tmp_file_path, file.filename)
            
#             # Generate summary
#             logger.info("Generating summary...")
#             summary = summarization_service.generate_summary(chunks)
            
#             # Store document data
#             document_id = f"doc_{len(document_store) + 1}"
#             document_store[document_id] = {
#                 'filename': file.filename,
#                 'chunks': chunks,
#                 'summary': summary,
#                 'full_text': ' '.join([chunk['text'] for chunk in chunks])
#             }
            
#             logger.info(f"Document processed successfully. ID: {document_id}")
            
#             return DocumentResponse(
#                 document_id=document_id,
#                 filename=file.filename,
#                 summary=summary,
#                 chunk_count=len(chunks)
#             )
            
#         finally:
#             # Clean up temporary file
#             os.unlink(tmp_file_path)
            
#     except Exception as e:
#         logger.error(f"Error processing document: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")






# @app.post("/answer", response_model=AnswerResponse)
# async def answer_question(request: QuestionRequest):
#     """Answer a question about the uploaded document"""
#     try:
#         if request.document_id not in document_store:
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         document_data = document_store[request.document_id]
        
#         logger.info(f"Answering question for document {request.document_id}: {request.question}")
        
#         # Get answer from QA service
#         answer_result = qa_service.answer_question(
#             question=request.question,
#             chunks=document_data['chunks']
#         )
        
#         return AnswerResponse(
#             answer=answer_result['answer'],
#             confidence=answer_result['confidence'],
#             source_reference=answer_result['source_reference']
#         )
        
#     except Exception as e:
#         logger.error(f"Error answering question: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")







# @app.post("/challenge", response_model=ChallengeResponse)
# async def generate_challenge(document_id: str):
#     """Generate challenge questions for the document"""
#     try:
#         if document_id not in document_store:
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         document_data = document_store[document_id]
        
#         logger.info(f"Generating challenge questions for document {document_id}")
        
#         # Generate challenge questions
#         questions = challenge_service.generate_questions(document_data['chunks'])
        
#         # Store questions for evaluation
#         if 'challenge_questions' not in document_data:
#             document_data['challenge_questions'] = {}
        
#         challenge_questions = []
#         for i, q in enumerate(questions):
#             question_id = i + 1
#             document_data['challenge_questions'][question_id] = q
#             challenge_questions.append(ChallengeQuestion(id=question_id, question=q['question']))
        
#         return ChallengeResponse(questions=challenge_questions)
        
#     except Exception as e:
#         logger.error(f"Error generating challenge: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error generating challenge: {str(e)}")






# @app.post("/evaluate", response_model=EvaluationResponse)
# async def evaluate_answer(request: ChallengeAnswerRequest):
#     """Evaluate user's answer to a challenge question"""
#     try:
#         if request.document_id not in document_store:
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         document_data = document_store[request.document_id]
        
#         if 'challenge_questions' not in document_data or request.question_id not in document_data['challenge_questions']:
#             raise HTTPException(status_code=404, detail="Challenge question not found")
        
#         question_data = document_data['challenge_questions'][request.question_id]
        
#         logger.info(f"Evaluating answer for question {request.question_id}")
        
#         # Evaluate answer
#         evaluation = challenge_service.evaluate_answer(
#             question_data=question_data,
#             user_answer=request.answer,
#             document_chunks=document_data['chunks']
#         )
        
#         return EvaluationResponse(
#             score=evaluation['score'],
#             feedback=evaluation['feedback'],
#             correct_answer=evaluation['correct_answer']
#         )
        
#     except Exception as e:
#         logger.error(f"Error evaluating answer: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")
    



# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy"}



# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)







from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile
from typing import List, Dict, Any
import logging

# Import from the merged summarizer.py
from services.summarizer import DocumentProcessor, SummarizationService
from services.qa_service import QAService
from services.challenge_service import ChallengeService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Research Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
doc_processor = DocumentProcessor()
summarization_service = SummarizationService()
qa_service = QAService()
challenge_service = ChallengeService()

# Store for uploaded documents (in production, use a database)
document_store = {}

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class ChallengeAnswerRequest(BaseModel):
    document_id: str
    question_id: int
    answer: str

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    summary: str
    chunk_count: int

class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    source_reference: str

class ChallengeQuestion(BaseModel):
    id: int
    question: str

class ChallengeResponse(BaseModel):
    questions: List[ChallengeQuestion]

class EvaluationResponse(BaseModel):
    score: int
    feedback: str
    correct_answer: str

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF or TXT)"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            logger.info(f"Processing document: {file.filename}")
            chunks = doc_processor.process_document(tmp_file_path, file.filename)
            
            # Generate summary
            logger.info("Generating summary...")
            summary = summarization_service.generate_summary(chunks)
            
            # Store document data
            document_id = f"doc_{len(document_store) + 1}"
            document_store[document_id] = {
                'filename': file.filename,
                'chunks': chunks,
                'summary': summary,
                'full_text': ' '.join([chunk['text'] for chunk in chunks])
            }
            
            logger.info(f"Document processed successfully. ID: {document_id}")
            
            return DocumentResponse(
                document_id=document_id,
                filename=file.filename,
                summary=summary,
                chunk_count=len(chunks)
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Answer a question about the uploaded document"""
    try:
        if request.document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document_data = document_store[request.document_id]
        
        logger.info(f"Answering question for document {request.document_id}: {request.question}")
        
        # Get answer from QA service
        answer_result = qa_service.answer_question(
            question=request.question,
            chunks=document_data['chunks']
        )
        
        return AnswerResponse(
            answer=answer_result['answer'],
            confidence=answer_result['confidence'],
            source_reference=answer_result['source_reference']
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/challenge", response_model=ChallengeResponse)
async def generate_challenge(document_id: str):
    """Generate challenge questions for the document"""
    try:
        if document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document_data = document_store[document_id]
        
        logger.info(f"Generating challenge questions for document {document_id}")
        
        # Generate challenge questions
        questions = challenge_service.generate_questions(document_data['chunks'])
        
        # Store questions for evaluation
        if 'challenge_questions' not in document_data:
            document_data['challenge_questions'] = {}
        
        challenge_questions = []
        for i, q in enumerate(questions):
            question_id = i + 1
            document_data['challenge_questions'][question_id] = q
            challenge_questions.append(ChallengeQuestion(id=question_id, question=q['question']))
        
        return ChallengeResponse(questions=challenge_questions)
        
    except Exception as e:
        logger.error(f"Error generating challenge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating challenge: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(request: ChallengeAnswerRequest):
    """Evaluate user's answer to a challenge question"""
    try:
        if request.document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document_data = document_store[request.document_id]
        
        if 'challenge_questions' not in document_data or request.question_id not in document_data['challenge_questions']:
            raise HTTPException(status_code=404, detail="Challenge question not found")
        
        question_data = document_data['challenge_questions'][request.question_id]
        
        logger.info(f"Evaluating answer for question {request.question_id}")
        
        # Evaluate answer
        evaluation = challenge_service.evaluate_answer(
            question_data=question_data,
            user_answer=request.answer,
            document_chunks=document_data['chunks']
        )
        
        return EvaluationResponse(
            score=evaluation['score'],
            feedback=evaluation['feedback'],
            correct_answer=evaluation['correct_answer']
        )
        
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)