# Smart Research Assistant
## Overview

An intelligent document analysis system with:
- **Backend API**: FastAPI service for document processing, summarization, and question answering
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
  - **Question Answering**: Natural language questions about documents
  - **Challenge Mode**: Generated questions with evaluation

## System Architecture
smart-research-assistant/
├── backend/
│ ├── main.py # FastAPI application
│ ├── summarizer.py # Document processing & summarization
│ ├── qa_service.py # Question answering system
│ ├── challenge_service.py # Challenge generation
├── frontend/
│ └── app.py # Streamlit application
├── requirements.txt
└── README.md # This document

## SETUP

### Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

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