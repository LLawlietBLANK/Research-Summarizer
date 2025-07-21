# Smart Research Assistant
## Overview
This Smart Research Assistant is an AI-powered tool designed to help users quickly understand and interact with documents through summarization, question answering, and comprehension challenges. 
An intelligent document analysis system with:
- **Backend API**: FastAPI service for document processing, summarization, question answering and challenging mode
- **Frontend Interface**: Streamlit web app for user interaction

## Features

### Backend Services
- **Document Processing**
  - Using google/longt-tglobal-base
  - PDF/TXT file upload and text extraction
  - Chunking with metadata preservation
  - FAISS vector embeddings for semantic search

### Frontend Interface
- **Document Management**
  - File upload with automatic processing
  - Document summary display
  - Clear document function

- **Interaction Modes**
  - **Question Answering**: Allows the user to ask questions related to the document and answers those questions based on the document, also providing the reference of the answer from the text.
  - **Challenge Mode**: Generated questions and challenges the user to answer those questions

## System Architecture
research-summarizer/

-backend/

  -main.py

  -services/

    -summarization_serivce.py

    -document_processor.py

    -qa_service.py

    -challenge-service.py

    -vector_store.py

-frontend/

  -app.py

-README.md

-requirements.txt


## SETUP

### Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS

.venv\Scripts\activate     # Windows

### Install dependencies
pip install -r requirements.txt

### Run backend
python main.py

### Run frontend
streamlit run app.py


## USAGE
1. Document Upload
Click "Browse files" to upload a PDF or TXT document

The system automatically processes the file and displays the summary

2. Interaction Modes
Question Answering Mode
Select "Ask Questions" mode

Type your question in the text box

Click "Get Answer" to receive response with confidence score

Challenge Mode
Select "Challenge Mode"

Click "Generate Questions"

Answer each question and submit for evaluation

View your score and feedback

3. Document Management
Use "Clear Document" to remove current file

Upload new files at any time