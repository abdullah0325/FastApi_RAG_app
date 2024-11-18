import streamlit as st
import requests
import os
from streamlit_chat import message
from dotenv import load_dotenv

# Configuration
BACKEND_URL = "http://localhost:8000"  # FastAPI backend URL

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your document")
    st.header("Abdullah GPT")

    # Initialize session state
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Sidebar for file upload and configuration
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your document", type=['pdf'], accept_multiple_files=True)
        blog_url = st.text_input("Enter the blog URL")
        youtube_url = st.text_input("Enter the YouTube video URL")
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        process = st.button("Submit")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        try:
            # Prepare the files and data for the API request
            files = []
            if uploaded_files:
                for file in uploaded_files:
                    files.append(("files", file))

            # Make API request to process the input
            response = requests.post(
                f"{BACKEND_URL}/process",
                files=files if files else None,
                data={
                    "blog_url": blog_url if blog_url else None,
                    "youtube_url": youtube_url if youtube_url else None,
                    "openai_api_key": openai_api_key
                }
            )
            
            if response.status_code == 200:
                st.session_state.conversation_id = response.json()["conversation_id"]
                st.session_state.processComplete = True
                st.write("Processing complete! You can now ask questions.")
            else:
                st.error(f"Error: {response.json()['detail']}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Chat interface
    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about the uploaded document, blog, or YouTube video.")
        if user_question:
            try:
                # Send question to backend
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "question": user_question,
                        "conversation_id": st.session_state.conversation_id
                    }
                )
                
                if response.status_code == 200:
                    # Update chat history
                    result = response.json()
                    st.session_state.chat_history = result["chat_history"]
                    
                    # Display chat history
                    response_container = st.container()
                    with response_container:
                        for message_obj in st.session_state.chat_history:
                            message(
                                message_obj["content"],
                                is_user=message_obj["role"] == "user",
                                key=str(hash(message_obj["content"]))
                            )
                else:
                    st.error(f"Error: {response.json()['detail']}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()