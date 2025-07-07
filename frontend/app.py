import streamlit as st
import requests
from datetime import datetime
import time

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Configure the page
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'document_id' not in st.session_state:
    st.session_state.document_id = None
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None
if 'document_filename' not in st.session_state:
    st.session_state.document_filename = None
if 'challenge_questions' not in st.session_state:
    st.session_state.challenge_questions = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'challenge_answers' not in st.session_state:
    st.session_state.challenge_answers = {}
if 'challenge_mode' not in st.session_state:
    st.session_state.challenge_mode = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = None

# API call functions with error handling
def upload_document(uploaded_file):
    """Upload document to backend"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with st.spinner("Processing document... This may take a minute"):
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error processing document: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Please ensure the backend is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def ask_question(question: str):
    """Ask a question about the document"""
    try:
        payload = {"document_id": st.session_state.document_id, "question": question}
        with st.spinner("Finding answer..."):
            response = requests.post(f"{BACKEND_URL}/answer", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting answer: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error communicating with server: {str(e)}")
        return None

def generate_challenge():
    """Generate challenge questions"""
    try:
        with st.spinner("Generating challenge questions..."):
            response = requests.post(f"{BACKEND_URL}/challenge?document_id={st.session_state.document_id}")
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.challenge_questions = result['questions']
            st.session_state.current_question_index = 0
            st.session_state.challenge_answers = {}
            st.session_state.challenge_mode = True
            return result
        else:
            st.error(f"Error generating questions: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None

def evaluate_answer(question_id: int, answer: str):
    """Evaluate user's answer to a challenge question"""
    try:
        payload = {
            "document_id": st.session_state.document_id,
            "question_id": question_id,
            "answer": answer
        }
        with st.spinner("Evaluating your answer..."):
            response = requests.post(f"{BACKEND_URL}/evaluate", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error evaluating answer: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error evaluating answer: {str(e)}")
        return None

def display_ask_anything_mode():
    """Display the Ask Anything mode interface"""
    st.header("❓ Ask Anything")
    st.markdown("Ask any question about your document and get AI-powered answers with source references.")
    
    # Question input
    question = st.text_area(
        "Your Question:",
        placeholder="e.g., What is the main conclusion of this document?",
        height=100,
        help="Ask any question about the content of your document"
    )
    
    if st.button("🔍 Get Answer", type="primary") and question:
        answer = ask_question(question)
        
        if answer:
            # Display answer
            st.subheader("💡 Answer")
            st.success(answer['answer'])
            
            # Confidence indicator
            confidence = answer['confidence']
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.3 else "red"
            confidence_text = f"**Confidence:** :{confidence_color}[{confidence:.1%}]"
            
            # Source reference
            source_ref = answer.get('source_reference', 'Page not specified')
            
            # Create columns for metrics
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(confidence_text)
            with col2:
                st.markdown(f"**Source Reference:** {source_ref}")
            
            # Confidence interpretation
            if confidence > 0.7:
                st.success("🎯 High confidence answer based on document content")
            elif confidence > 0.3:
                st.warning("⚠️ Medium confidence - verify with document if critical")
            else:
                st.error("❌ Low confidence - answer may not be reliable")
        else:
            st.error("Could not get an answer. Please try again with a different question.")

def display_challenge_mode():
    """Display the Challenge Mode interface"""
    st.header("🏆 Challenge Mode")
    st.markdown("Test your understanding with AI-generated questions about your document.")
    
    if not st.session_state.challenge_questions:
        st.info("No challenge questions available yet. Click below to generate them.")
        if st.button("Generate Challenge Questions", type="primary"):
            generate_challenge()
        return
    
    # Progress indicator
    total_questions = len(st.session_state.challenge_questions)
    current_q = st.session_state.current_question_index
    
    # Progress bar and counter
    progress = (current_q) / total_questions
    st.progress(progress)
    st.caption(f"Question {current_q + 1} of {total_questions}")
    
    # Current question
    if current_q < total_questions:
        question = st.session_state.challenge_questions[current_q]
        question_id = question['id']
        
        # Display question
        st.subheader(f"🤔 Question {current_q + 1}")
        st.markdown(f"**{question['question']}**")
        
        # Answer input
        answer_key = f"answer_{question_id}"
        user_answer = st.text_area(
            "Your Answer:",
            key=answer_key,
            placeholder="Type your answer here...",
            height=150,
            help="Provide a comprehensive answer based on the document content"
        )
        
        # Navigation and submit buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📝 Submit Answer", type="primary", disabled=not user_answer):
                evaluation = evaluate_answer(question_id, user_answer)
                
                if evaluation:
                    st.session_state.challenge_answers[question_id] = {
                        'user_answer': user_answer,
                        'evaluation': evaluation
                    }
                    st.rerun()
        
        with col2:
            if current_q > 0 and st.button("⬅️ Previous"):
                st.session_state.current_question_index -= 1
                st.rerun()
        
        with col3:
            if current_q < total_questions - 1 and st.button("➡️ Next"):
                st.session_state.current_question_index += 1
                st.rerun()
        
        # Display evaluation if available
        if question_id in st.session_state.challenge_answers:
            evaluation = st.session_state.challenge_answers[question_id]['evaluation']
            
            st.divider()
            st.subheader("📊 Evaluation")
            
            # Score display
            score = evaluation['score']
            if score == 1:
                st.success("✅ Correct Answer!")
            else:
                st.error("❌ Incorrect Answer")
            
            # Feedback
            st.markdown("**Feedback:**")
            st.info(evaluation['feedback'])
            
            # Correct answer
            with st.expander("💡 See Expected Answer"):
                st.markdown(evaluation['correct_answer'])
    
    else:
        # Challenge completed
        st.success("🎉 Challenge Completed!")
        
        # Calculate overall score
        total_score = 0
        answered_questions = 0
        
        for q_id, answer_data in st.session_state.challenge_answers.items():
            total_score += answer_data['evaluation']['score']
            answered_questions += 1
        
        if answered_questions > 0:
            score_percentage = (total_score / answered_questions) * 100
            st.subheader(f"🏆 Your Score: {score_percentage:.0f}%")
            
            # Display performance message
            if score_percentage >= 80:
                st.success("Excellent! You have a deep understanding of the document.")
            elif score_percentage >= 60:
                st.success("Good job! You understand the main points.")
            elif score_percentage >= 40:
                st.warning("Fair understanding. Review the document again.")
            else:
                st.error("Needs improvement. Consider studying the document more carefully.")
        else:
            st.info("You didn't answer any questions.")
        
        if st.button("🔄 Restart Challenge", type="primary"):
            st.session_state.current_question_index = 0
            st.session_state.challenge_answers = {}
            st.rerun()

def main():
    st.title("🧠 Smart Research Assistant")
    st.markdown("Upload documents, get summaries, ask questions, and test your understanding with AI-generated challenges.")
    
    # Sidebar for document upload and info
    with st.sidebar:
        st.header("📄 Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Supported formats: PDF and plain text files"
        )
        

        if uploaded_file is not None and (st.session_state.last_uploaded != uploaded_file.name or st.session_state.document_filename != uploaded_file.name):
            st.session_state.processing = True
            st.session_state.last_uploaded = uploaded_file.name
            
            result = upload_document(uploaded_file)
            
            st.session_state.processing = False
            
            if result:
                st.session_state.document_id = result['document_id']
                st.session_state.document_summary = result['summary']
                st.session_state.document_filename = result['filename']
                st.session_state.challenge_questions = []
                st.session_state.challenge_answers = {}
                st.session_state.challenge_mode = False
                st.rerun()


        # if uploaded_file is not None:
        #     if st.button("Process Document", type="primary"):
        #         result = upload_document(uploaded_file)
                
        #         if result:
        #             st.session_state.document_id = result['document_id']
        #             st.session_state.document_summary = result['summary']
        #             st.session_state.document_filename = result['filename']
        #             st.session_state.challenge_questions = []
        #             st.session_state.challenge_answers = {}
        #             st.session_state.challenge_mode = False
        #             st.success(f"✅ Document processed successfully!")
        #             st.rerun()
        
        # Document info section
        if st.session_state.document_id:
            st.divider()
            st.subheader("Current Document")
            st.markdown(f"**📄 File:** {st.session_state.document_filename}")
            st.caption(f"**ID:** `{st.session_state.document_id}`")
            
            # Clear document button
            if st.button("🗑️ Clear Document", use_container_width=True):
                st.session_state.document_id = None
                st.session_state.document_summary = None
                st.session_state.document_filename = None
                st.session_state.challenge_questions = []
                st.session_state.challenge_answers = {}
                st.session_state.challenge_mode = False
                st.rerun()
        
        # How to use section
        st.divider()
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Upload** a PDF or text document
            2. **Review** the auto-generated summary
            3. Choose a mode:
               - **Ask Anything**: Ask questions about the document
               - **Challenge Me**: Test your understanding with AI-generated questions
            4. **Submit answers** and get immediate feedback
            """)
    
    # Main content area
    if not st.session_state.document_id:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 5rem 1rem;'>
            <h2 style='margin-bottom: 1.5rem;'>Welcome to Smart Research Assistant!</h2>
            <p style='font-size: 1.2rem; max-width: 700px; margin: 0 auto 2rem;'>
            This tool helps you quickly understand documents through AI-powered summarization,
            question answering, and comprehension challenges.
            </p>
            <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 3rem;'>
                <div style='background: #f0f2f6; border-radius: 10px; padding: 1.5rem; width: 250px;'>
                    <h3>📄 Document Upload</h3>
                    <p>Upload PDF or text files for processing</p>
                </div>
                <div style='background: #f0f2f6; border-radius: 10px; padding: 1.5rem; width: 250px;'>
                    <h3>📝 Auto Summary</h3>
                    <p>Get concise 150-word summaries instantly</p>
                </div>
                <div style='background: #f0f2f6; border-radius: 10px; padding: 1.5rem; width: 250px;'>
                    <h3>❓ Ask Anything</h3>
                    <p>Get answers to your questions with source references</p>
                </div>
                <div style='background: #f0f2f6; border-radius: 10px; padding: 1.5rem; width: 250px;'>
                    <h3>🏆 Challenge Me</h3>
                    <p>Test your understanding with AI-generated questions</p>
                </div>
            </div>
            <p style='margin-top: 3rem; font-size: 1.1rem;'>
            <strong>Get started by uploading a document in the sidebar!</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Display document summary
        st.header("📋 Document Summary")
        st.markdown(f"**📄 {st.session_state.document_filename}**")
        
        with st.container():
            st.markdown(st.session_state.document_summary)
        
        # Mode selection
        st.header("🎯 Interaction Mode")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("❓ Ask Anything Mode", use_container_width=True, 
                         help="Ask free-form questions about the document"):
                st.session_state.challenge_mode = False
                st.rerun()
        
        with col2:
            if st.button("🏆 Challenge Me Mode", use_container_width=True,
                         help="Test your understanding with AI-generated questions"):
                if not st.session_state.challenge_questions:
                    generate_challenge()
                else:
                    st.session_state.challenge_mode = True
                    st.rerun()
        
        # Display selected mode
        st.divider()
        if st.session_state.challenge_mode:
            display_challenge_mode()
        else:
            display_ask_anything_mode()

if __name__ == "__main__":
    main()