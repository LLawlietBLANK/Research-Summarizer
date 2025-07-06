# import streamlit as st
# import requests
# import json
# from typing import Dict, Any, List
# import time

# # Configure the page
# st.set_page_config(
#     page_title="Smart Research Assistant",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Backend API URL
# BACKEND_URL = "http://localhost:8000"

# # Initialize session state
# if 'document_id' not in st.session_state:
#     st.session_state.document_id = None
# if 'document_summary' not in st.session_state:
#     st.session_state.document_summary = None
# if 'document_filename' not in st.session_state:
#     st.session_state.document_filename = None
# if 'challenge_questions' not in st.session_state:
#     st.session_state.challenge_questions = []
# if 'current_question_index' not in st.session_state:
#     st.session_state.current_question_index = 0
# if 'challenge_answers' not in st.session_state:
#     st.session_state.challenge_answers = {}
# if 'challenge_mode' not in st.session_state:
#     st.session_state.challenge_mode = False

# def upload_document(uploaded_file):
#     """Upload document to backend"""
#     try:
#         files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
#         with st.spinner("Processing document... This may take a few minutes."):
#             response = requests.post(f"{BACKEND_URL}/upload", files=files)
        
#         if response.status_code == 200:
#             result = response.json()
#             st.session_state.document_id = result['document_id']
#             st.session_state.document_summary = result['summary']
#             st.session_state.document_filename = result['filename']
#             st.session_state.challenge_questions = []
#             st.session_state.challenge_answers = {}
#             st.session_state.challenge_mode = False
#             return True, result
#         else:
#             return False, f"Error: {response.status_code} - {response.text}"
    
#     except requests.exceptions.ConnectionError:
#         return False, "Cannot connect to backend server. Please ensure the backend is running on localhost:8000"
#     except Exception as e:
#         return False, f"An error occurred: {str(e)}"

# def ask_question(question: str):
#     """Ask a question about the document"""
#     try:
#         payload = {
#             "document_id": st.session_state.document_id,
#             "question": question
#         }
        
#         response = requests.post(f"{BACKEND_URL}/answer", json=payload)
        
#         if response.status_code == 200:
#             return True, response.json()
#         else:
#             return False, f"Error: {response.status_code} - {response.text}"
    
#     except Exception as e:
#         return False, f"An error occurred: {str(e)}"

# def generate_challenge():
#     """Generate challenge questions"""
#     try:
#         response = requests.post(f"{BACKEND_URL}/challenge?document_id={st.session_state.document_id}")
        
#         if response.status_code == 200:
#             result = response.json()
#             st.session_state.challenge_questions = result['questions']
#             st.session_state.current_question_index = 0
#             st.session_state.challenge_answers = {}
#             st.session_state.challenge_mode = True
#             return True, result
#         else:
#             return False, f"Error: {response.status_code} - {response.text}"
    
#     except Exception as e:
#         return False, f"An error occurred: {str(e)}"

# def evaluate_answer(question_id: int, answer: str):
#     """Evaluate user's answer to a challenge question"""
#     try:
#         payload = {
#             "document_id": st.session_state.document_id,
#             "question_id": question_id,
#             "answer": answer
#         }
        
#         response = requests.post(f"{BACKEND_URL}/evaluate", json=payload)
        
#         if response.status_code == 200:
#             return True, response.json()
#         else:
#             return False, f"Error: {response.status_code} - {response.text}"
    
#     except Exception as e:
#         return False, f"An error occurred: {str(e)}"

# def main():
#     st.title("🧠 Smart Research Assistant")
#     st.markdown("Upload a document and get AI-powered summaries and interactive Q&A")
    
#     # Sidebar for document upload
#     with st.sidebar:
#         st.header("📄 Document Upload")
        
#         uploaded_file = st.file_uploader(
#             "Choose a PDF or TXT file",
#             type=['pdf', 'txt'],
#             help="Upload a document to get started with summarization and Q&A"
#         )
        
#         if uploaded_file is not None:
#             if st.button("Process Document", type="primary"):
#                 success, result = upload_document(uploaded_file)
                
#                 if success:
#                     st.success(f"✅ Document '{result['filename']}' processed successfully!")
#                     st.info(f"📊 Created {result['chunk_count']} text chunks")
#                     st.rerun()
#                 else:
#                     st.error(f"❌ {result}")
        
#         # Document info
#         if st.session_state.document_id:
#             st.markdown("---")
#             st.markdown("**📋 Current Document:**")
#             st.info(f"📄 {st.session_state.document_filename}")
#             st.markdown(f"**ID:** `{st.session_state.document_id}`")
            
#             if st.button("🔄 Clear Document"):
#                 st.session_state.document_id = None
#                 st.session_state.document_summary = None
#                 st.session_state.document_filename = None
#                 st.session_state.challenge_questions = []
#                 st.session_state.challenge_answers = {}
#                 st.session_state.challenge_mode = False
#                 st.rerun()
    
#     # Main content area
#     if st.session_state.document_id is None:
#         st.markdown("""
#         ## 🚀 Welcome to Smart Research Assistant!
        
#         This tool helps you:
#         - 📄 **Upload** PDF or TXT documents
#         - 📝 **Generate** automatic summaries
#         - ❓ **Ask questions** about your documents
#         - 🎯 **Test comprehension** with AI-generated challenges
        
#         **Get started by uploading a document in the sidebar!**
#         """)
        
#         # Example usage
#         with st.expander("🔍 How to Use"):
#             st.markdown("""
#             1. **Upload Document**: Use the sidebar to upload a PDF or TXT file
#             2. **Review Summary**: Get an automatic 150-word summary
#             3. **Ask Questions**: Use "Ask Anything" mode to query your document
#             4. **Take Challenge**: Test your understanding with AI-generated questions
#             """)
    
#     else:
#         # Display document summary
#         st.header("📋 Document Summary")
        
#         if st.session_state.document_summary:
#             st.markdown(f"**📄 {st.session_state.document_filename}**")
            
#             with st.container():
#                 st.markdown("### 📝 Auto-Generated Summary")
#                 st.info(st.session_state.document_summary)
        
#         # Mode selection
#         st.header("🎯 Choose Your Mode")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if st.button("❓ Ask Anything Mode", use_container_width=True):
#                 st.session_state.challenge_mode = False
#                 st.rerun()
        
#         with col2:
#             if st.button("🏆 Challenge Me Mode", use_container_width=True):
#                 if not st.session_state.challenge_questions:
#                     with st.spinner("Generating challenge questions..."):
#                         success, result = generate_challenge()
                    
#                     if success:
#                         st.success("✅ Challenge questions generated!")
#                         st.rerun()
#                     else:
#                         st.error(f"❌ {result}")
#                 else:
#                     st.session_state.challenge_mode = True
#                     st.rerun()
        
#         # Display selected mode
#         if st.session_state.challenge_mode:
#             display_challenge_mode()
#         else:
#             display_ask_anything_mode()

# def display_ask_anything_mode():
#     """Display the Ask Anything mode interface"""
#     st.header("❓ Ask Anything Mode")
#     st.markdown("Ask any question about your document and get AI-powered answers with source references.")
    
#     # Question input
#     question = st.text_input(
#         "Your Question:",
#         placeholder="e.g., What is the main conclusion of this document?",
#         help="Ask any question about the content of your document"
#     )
    
#     if st.button("🔍 Get Answer", type="primary") and question:
#         with st.spinner("Finding answer..."):
#             success, result = ask_question(question)
        
#         if success:
#             st.markdown("### 💡 Answer")
            
#             # Display answer
#             st.success(result['answer'])
            
#             # Display confidence and source
#             col1, col2 = st.columns(2)
#             with col1:
#                 confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.3 else "red"
#                 st.markdown(f"**Confidence:** :{confidence_color}[{result['confidence']:.1%}]")
            
#             with col2:
#                 st.markdown("**Source Reference:**")
#                 st.caption(result['source_reference'])
            
#             # Confidence interpretation
#             if result['confidence'] > 0.7:
#                 st.success("🎯 High confidence answer")
#             elif result['confidence'] > 0.3:
#                 st.warning("⚠️ Medium confidence - verify if needed")
#             else:
#                 st.error("❌ Low confidence - answer may not be reliable")
        
#         else:
#             st.error(f"❌ {result}")

# def display_challenge_mode():
#     """Display the Challenge Mode interface"""
#     st.header("🏆 Challenge Me Mode")
#     st.markdown("Test your understanding with AI-generated questions about your document.")
    
#     if not st.session_state.challenge_questions:
#         st.warning("No challenge questions available. Click 'Challenge Me Mode' to generate them.")
#         return
    
#     # Progress indicator
#     total_questions = len(st.session_state.challenge_questions)
#     current_q = st.session_state.current_question_index
    
#     st.progress((current_q) / total_questions)
#     st.markdown(f"**Question {current_q + 1} of {total_questions}**")
    
#     # Current question
#     if current_q < total_questions:
#         question = st.session_state.challenge_questions[current_q]
#         question_id = question['id']
        
#         st.markdown(f"### 🤔 {question['question']}")
        
#         # Answer input
#         answer_key = f"answer_{question_id}"
#         user_answer = st.text_area(
#             "Your Answer:",
#             key=answer_key,
#             placeholder="Type your answer here...",
#             help="Provide a comprehensive answer based on the document content"
#         )
        
#         col1, col2, col3 = st.columns([1, 1, 1])
        
#         with col1:
#             if st.button("📝 Submit Answer", type="primary") and user_answer:
#                 with st.spinner("Evaluating your answer..."):
#                     success, result = evaluate_answer(question_id, user_answer)
                
#                 if success:
#                     st.session_state.challenge_answers[question_id] = {
#                         'user_answer': user_answer,
#                         'evaluation': result
#                     }
#                     st.rerun()
#                 else:
#                     st.error(f"❌ {result}")
        
#         with col2:
#             if current_q > 0:
#                 if st.button("⬅️ Previous"):
#                     st.session_state.current_question_index -= 1
#                     st.rerun()
        
#         with col3:
#             if current_q < total_questions - 1:
#                 if st.button("➡️ Next"):
#                     st.session_state.current_question_index += 1
#                     st.rerun()
        
#         # Display evaluation if available
#         if question_id in st.session_state.challenge_answers:
#             evaluation = st.session_state.challenge_answers[question_id]['evaluation']
            
#             st.markdown("---")
#             st.markdown("### 📊 Evaluation Results")
            
#             # Score display
#             score = evaluation['score']
#             if score >= 8:
#                 st.success(f"🎉 Excellent! Score: {score}/10")
#             elif score >= 6:
#                 st.success(f"👍 Good! Score: {score}/10")
#             elif score >= 4:
#                 st.warning(f"⚠️ Fair: Score: {score}/10")
#             else:
#                 st.error(f"❌ Needs improvement: Score: {score}/10")
            
#             # Feedback
#             st.markdown("**Feedback:**")
#             st.info(evaluation['feedback'])
            
#             # Correct answer
#             with st.expander("💡 See Expected Answer"):
#                 st.markdown(evaluation['correct_answer'])
    
#     else:
#         # Challenge completed
#         st.success("🎉 Challenge Completed!")
        
#         # Calculate overall score
#         total_score = 0
#         answered_questions = 0
        
#         for q_id, answer_data in st.session_state.challenge_answers.items():
#             total_score += answer_data['evaluation']['score']
#             answered_questions += 1
        
#         if answered_questions > 0:
#             avg_score = total_score / answered_questions
#             st.markdown(f"### 🏆 Your Average Score: {avg_score:.1f}/10")
#         else:
#             st.markdown("### 🏆 You didn't answer any questions.")
#         st.markdown("**Thank you for participating!**")


# if __name__ == "__main__":
#     main()
#     # Run the Streamlit app
#     # Use `streamlit run app.py` in terminal to start the app



























# import streamlit as st
# import requests
# from typing import Dict, Any

# # Backend API URL
# BACKEND_URL = "http://localhost:8000"

# # Configure the page
# st.set_page_config(
#     page_title="Research Assistant",
#     page_icon="📚",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )

# # Initialize session state
# if 'document_id' not in st.session_state:
#     st.session_state.document_id = None
# if 'document_summary' not in st.session_state:
#     st.session_state.document_summary = None
# if 'document_filename' not in st.session_state:
#     st.session_state.document_filename = None
# if 'challenge_questions' not in st.session_state:
#     st.session_state.challenge_questions = []
# if 'current_question_index' not in st.session_state:
#     st.session_state.current_question_index = 0
# if 'challenge_answers' not in st.session_state:
#     st.session_state.challenge_answers = {}
# if 'challenge_mode' not in st.session_state:
#     st.session_state.challenge_mode = False

# def upload_document(uploaded_file):
#     """Upload document to backend"""
#     try:
#         files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
#         with st.spinner("Processing..."):
#             response = requests.post(f"{BACKEND_URL}/upload", files=files)
#         return response.json() if response.status_code == 200 else None
#     except requests.exceptions.ConnectionError:
#         st.error("Cannot connect to backend server")
#         return None

# def ask_question(question: str):
#     """Ask a question about the document"""
#     try:
#         payload = {"document_id": st.session_state.document_id, "question": question}
#         response = requests.post(f"{BACKEND_URL}/answer", json=payload)
#         return response.json() if response.status_code == 200 else None
#     except Exception:
#         return None

# def generate_challenge():
#     """Generate challenge questions"""
#     try:
#         response = requests.post(f"{BACKEND_URL}/challenge?document_id={st.session_state.document_id}")
#         if response.status_code == 200:
#             result = response.json()
#             st.session_state.challenge_questions = result['questions']
#             st.session_state.current_question_index = 0
#             st.session_state.challenge_answers = {}
#             st.session_state.challenge_mode = True
#             return result
#         return None
#     except Exception:
#         return None

# def evaluate_answer(question_id: int, answer: str):
#     """Evaluate user's answer to a challenge question"""
#     try:
#         payload = {
#             "document_id": st.session_state.document_id,
#             "question_id": question_id,
#             "answer": answer
#         }
#         response = requests.post(f"{BACKEND_URL}/evaluate", json=payload)
#         return response.json() if response.status_code == 200 else None
#     except Exception:
#         return None

# def show_question_answer_mode():
#     """Show the Q&A interface"""
#     st.subheader("Ask about the document")
#     question = st.text_input("Your question", key="question_input")
    
#     if st.button("Get Answer", type="primary") and question:
#         answer = ask_question(question)
#         if answer:
#             with st.expander("Answer", expanded=True):
#                 st.markdown(answer['answer'])
#                 st.caption(f"Confidence: {answer['confidence']:.0%}")
#                 if answer['confidence'] < 0.5:
#                     st.warning("This answer has lower confidence")

# def show_challenge_mode():
#     """Show the challenge interface"""
#     st.subheader("Challenge Mode")
    
#     if not st.session_state.challenge_questions:
#         if st.button("Generate Questions", type="primary"):
#             result = generate_challenge()
#             if not result:
#                 st.error("Failed to generate questions")
#         return
    
#     total = len(st.session_state.challenge_questions)
#     current = st.session_state.current_question_index
#     question = st.session_state.challenge_questions[current]
    
#     # Progress
#     st.caption(f"Question {current+1} of {total}")
    
#     # Question
#     st.markdown(f"**{question['question']}**")
    
#     # Answer input
#     answer_key = f"answer_{question['id']}"
#     user_answer = st.text_area("Your answer", key=answer_key)
    
#     # Navigation
#     col1, col2 = st.columns(2)
#     with col1:
#         if current > 0 and st.button("Previous"):
#             st.session_state.current_question_index -= 1
#             st.rerun()
#     with col2:
#         if current < total-1 and st.button("Next"):
#             st.session_state.current_question_index += 1
#             st.rerun()
    
#     # Submit
#     if user_answer and st.button("Submit Answer", type="primary"):
#         evaluation = evaluate_answer(question['id'], user_answer)
#         if evaluation:
#             st.session_state.challenge_answers[question['id']] = {
#                 'user_answer': user_answer,
#                 'evaluation': evaluation
#             }
#             st.rerun()
    
#     # Show evaluation if exists
#     if question['id'] in st.session_state.challenge_answers:
#         eval_data = st.session_state.challenge_answers[question['id']]['evaluation']
#         st.divider()
#         st.markdown(f"**Score:** {eval_data['score']}/10")
#         st.markdown(f"**Feedback:** {eval_data['feedback']}")
#         with st.expander("Expected Answer"):
#             st.markdown(eval_data['correct_answer'])

# def main():
#     st.title("📚 Research Assistant")
    
#     # Document upload
#     uploaded_file = st.file_uploader("Upload document (PDF/TXT)", type=['pdf', 'txt'])
    
#     if uploaded_file and st.button("Process Document", type="primary"):
#         result = upload_document(uploaded_file)
#         if result:
#             st.session_state.document_id = result['document_id']
#             st.session_state.document_summary = result['summary']
#             st.session_state.document_filename = result['filename']
#             st.success(f"Ready: {result['filename']}")
#         else:
#             st.error("Processing failed")

#     # Document interaction
#     if st.session_state.document_id:
#         st.divider()
        
#         # Document info
#         with st.expander(f"📄 {st.session_state.document_filename}"):
#             st.write(st.session_state.document_summary)
        
#         # Mode selection
#         mode = st.radio("Select mode:", 
#                        ["Ask Questions", "Challenge Mode"],
#                        horizontal=True)
        
#         if mode == "Ask Questions":
#             st.session_state.challenge_mode = False
#             show_question_answer_mode()
#         else:
#             st.session_state.challenge_mode = True
#             show_challenge_mode()
        
#         if st.button("Clear Document"):
#             st.session_state.document_id = None
#             st.session_state.document_summary = None
#             st.session_state.document_filename = None
#             st.session_state.challenge_questions = []
#             st.session_state.challenge_answers = {}
#             st.session_state.challenge_mode = False
#             st.rerun()

# if __name__ == "__main__":
#     main()





import streamlit as st
import requests
from typing import Dict, Any

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Configure the page
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
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

def upload_document(uploaded_file):
    """Upload document to backend"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with st.spinner("Processing document..."):
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        st.error(f"Error processing document: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def ask_question(question: str):
    """Ask a question about the document"""
    try:
        payload = {"document_id": st.session_state.document_id, "question": question}
        response = requests.post(f"{BACKEND_URL}/answer", json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error asking question: {str(e)}")
        return None

def generate_challenge():
    """Generate challenge questions"""
    try:
        response = requests.post(f"{BACKEND_URL}/challenge?document_id={st.session_state.document_id}")
        if response.status_code == 200:
            result = response.json()
            st.session_state.challenge_questions = result['questions']
            st.session_state.current_question_index = 0
            st.session_state.challenge_answers = {}
            st.session_state.challenge_mode = True
            return result
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
        response = requests.post(f"{BACKEND_URL}/evaluate", json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error evaluating answer: {str(e)}")
        return None

def show_question_answer_mode():
    """Show the Q&A interface"""
    st.subheader("Ask about the document")
    question = st.text_input("Your question", key="question_input")
    
    if st.button("Get Answer", type="primary") and question:
        with st.spinner("Getting answer..."):
            answer = ask_question(question)
        if answer:
            with st.expander("Answer", expanded=True):
                st.markdown(answer['answer'])
                st.caption(f"Confidence: {answer['confidence']:.0%}")
                if answer['confidence'] < 0.5:
                    st.warning("This answer has lower confidence")

def show_challenge_mode():
    """Show the challenge interface"""
    st.subheader("Challenge Mode")
    
    if not st.session_state.challenge_questions:
        if st.button("Generate Questions", type="primary"):
            with st.spinner("Generating questions..."):
                result = generate_challenge()
            if not result:
                st.error("Failed to generate questions")
            else:
                st.rerun()
        return
    
    total = len(st.session_state.challenge_questions)
    current = st.session_state.current_question_index
    question = st.session_state.challenge_questions[current]
    
    # Progress
    st.caption(f"Question {current+1} of {total}")
    
    # Question
    st.markdown(f"**{question['question']}**")
    
    # Answer input
    answer_key = f"answer_{question['id']}"
    user_answer = st.text_area("Your answer", key=answer_key)
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if current > 0 and st.button("Previous"):
            st.session_state.current_question_index -= 1
            st.rerun()
    with col2:
        if current < total-1 and st.button("Next"):
            st.session_state.current_question_index += 1
            st.rerun()
    
    # Submit
    if user_answer and st.button("Submit Answer", type="primary"):
        with st.spinner("Evaluating answer..."):
            evaluation = evaluate_answer(question['id'], user_answer)
        if evaluation:
            st.session_state.challenge_answers[question['id']] = {
                'user_answer': user_answer,
                'evaluation': evaluation
            }
            st.rerun()
    
    # Show evaluation if exists
    if question['id'] in st.session_state.challenge_answers:
        eval_data = st.session_state.challenge_answers[question['id']]['evaluation']
        st.divider()
        st.markdown(f"**Score:** {eval_data['score']}/10")
        st.markdown(f"**Feedback:** {eval_data['feedback']}")
        with st.expander("Expected Answer"):
            st.markdown(eval_data['correct_answer'])

def main():
    st.title("📚 Research Assistant")
    
    # Document upload
    uploaded_file = st.file_uploader("Upload document (PDF/TXT)", type=['pdf', 'txt'])
    
    # Automatic processing when file is uploaded
    if uploaded_file and (st.session_state.document_filename != uploaded_file.name or st.session_state.document_id is None):
        st.session_state.processing = True
        result = upload_document(uploaded_file)
        st.session_state.processing = False
        
        if result:
            st.session_state.document_id = result['document_id']
            st.session_state.document_summary = result['summary']
            st.session_state.document_filename = result['filename']
            st.success(f"Document processed: {result['filename']}")
            st.rerun()
        else:
            st.session_state.document_id = None
            st.session_state.document_summary = None
            st.session_state.document_filename = None

    # Document interaction
    if st.session_state.document_id and not st.session_state.processing:
        st.divider()
        
        # Document info
        with st.expander(f"📄 {st.session_state.document_filename}"):
            st.write(st.session_state.document_summary)
        
        # Mode selection
        mode = st.radio("Select mode:", 
                       ["Ask Questions", "Challenge Mode"],
                       horizontal=True)
        
        if mode == "Ask Questions":
            st.session_state.challenge_mode = False
            show_question_answer_mode()
        else:
            st.session_state.challenge_mode = True
            show_challenge_mode()
        
        if st.button("Clear Document"):
            st.session_state.document_id = None
            st.session_state.document_summary = None
            st.session_state.document_filename = None
            st.session_state.challenge_questions = []
            st.session_state.challenge_answers = {}
            st.session_state.challenge_mode = False
            st.rerun()

if __name__ == "__main__":
    main()