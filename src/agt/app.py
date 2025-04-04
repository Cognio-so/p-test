import streamlit as st
import os
import uuid
import time
from agent import graph, VaaniState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
from typing import List, Dict, Any, Optional
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
def init_session_state():
    """Initialize or reset the session state variables."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o-mini"

# Handle file upload
def handle_file_upload(uploaded_file):
    """Process uploaded file and return the local file path."""
    if uploaded_file is None:
        return None
    try:
        timestamp = int(time.time())
        file_extension = os.path.splitext(uploaded_file.name)[1]
        unique_filename = f"{timestamp}_{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        abs_path = os.path.abspath(file_path)
        logger.info(f"File uploaded: {uploaded_file.name} -> {abs_path}")
        return abs_path
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        st.error(f"Failed to process the uploaded file: {str(e)}")
        return None

# Process user query
def process_query(query: str, model: str, file_path: Optional[str], deep_research: bool):
    """Process user query through the agent and update session state."""
    try:
        input_state = {
            "messages": st.session_state.messages + [HumanMessage(content=query)],
            "file_url": file_path,
            "indexed": st.session_state.indexed,
            "deep_research_requested": deep_research,
            "model_name": model
        }
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        logger.info(f"Processing query - Thread: {st.session_state.thread_id}, Model: {model}, Deep research: {deep_research}")
        with st.spinner("Thinking..."):
            result = graph.invoke(input_state, config)
        if "indexed" in result:
            st.session_state.indexed = result["indexed"]
        new_messages = result.get("messages", [])
        if new_messages:
            st.session_state.messages.append(HumanMessage(content=query))
            for msg in new_messages:
                if msg not in st.session_state.messages:
                    st.session_state.messages.append(msg)
        return True
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "apikey" in error_msg.lower():
            st.error("API key error: Please check your API keys in .env file")
            st.session_state.messages.append(HumanMessage(content=query))
            st.session_state.messages.append(AIMessage(content="I encountered an API key error. Please ensure all required API keys are properly configured."))
        elif "rate limit" in error_msg.lower():
            st.error("Rate limit exceeded. Please try again in a moment.")
            st.session_state.messages.append(HumanMessage(content=query))
            st.session_state.messages.append(AIMessage(content="I've hit a rate limit with one of the AI services. Please try again in a moment."))
        else:
            st.error(f"Error processing your request: {error_msg}")
            st.session_state.messages.append(HumanMessage(content=query))
            st.session_state.messages.append(AIMessage(content="I encountered an error while processing your query. Please try again with a different question."))
        return False

# Display chat messages
def display_chat_messages():
    """Display the conversation history in a chat-like interface."""
    for i, msg in enumerate(st.session_state.messages):
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                # Check if the message contains an image URL
                content = msg.content

                # Look for URLs that are likely images
                image_url_pattern = r'(https?://\S+\.(?:jpg|jpeg|png|gif|webp))'

                # Also check for Replicate image URL pattern
                replicate_url_pattern = r'(https?://\S+\.replicate\.\S+)'

                # Find all URLs in the message
                image_urls = re.findall(image_url_pattern, content, re.IGNORECASE)
                replicate_urls = re.findall(replicate_url_pattern, content, re.IGNORECASE)
                all_image_urls = image_urls + replicate_urls

                if all_image_urls:
                    # For each found image URL
                    for img_url in all_image_urls:
                        # Replace the URL in the content with an empty string or a placeholder
                        content = content.replace(img_url, "")

                    # Display the text content without the URLs
                    if content.strip():
                        st.write(content.strip())

                    # Display each image
                    for img_url in all_image_urls:
                        st.image(img_url, use_column_width=True)
                else:
                    # No image URLs found, display as normal text
                    st.write(content)

# Clear conversation
def clear_conversation():
    """Reset the conversation state while maintaining the thread ID."""
    try:
        thread_id = st.session_state.thread_id
        init_session_state()
        st.session_state.thread_id = thread_id
        logger.info(f"Conversation cleared for thread {thread_id}")
        st.success("Conversation cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        st.error(f"Error clearing conversation: {str(e)}")

# Main Streamlit app
def main():
    """Main Streamlit application logic."""
    try:
        st.set_page_config(page_title="Vaani.pro Chatbot", page_icon="🤖", layout="wide")
        required_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
            'TAVILY_API_KEY': os.getenv('TAVILY_API_KEY'),
            'QDRANT_URL': os.getenv('QDRANT_URL'),
            'QDRANT_API_KEY': os.getenv('QDRANT_API_KEY')
        }
        
        # Add optional keys with their descriptions
        optional_keys = {
            'REPLICATE_API_TOKEN': {
                'value': os.getenv('REPLICATE_API_TOKEN'),
                'description': 'Required for image generation'
            }
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        if missing_keys:
            st.error(f"Missing required API keys: {', '.join(missing_keys)}")
            st.warning("The application may not function correctly without these keys.")
            
        missing_optional = [f"{key} ({details['description']})" for key, details in optional_keys.items() if not details['value']]
        if missing_optional:
            st.warning(f"Missing optional API keys: {', '.join(missing_optional)}")
            st.info("Some features may be limited or unavailable.")
        init_session_state()
        with st.sidebar:
            st.title("Vaani.pro Settings")
            st.subheader("Conversation Thread")
            thread_id = st.text_input("Thread ID", value=st.session_state.thread_id)
            if thread_id != st.session_state.thread_id:
                st.session_state.thread_id = thread_id
                st.session_state.messages = []
                st.session_state.indexed = False
                logger.info(f"Switched to thread: {thread_id}")
            st.subheader("Model Selection")
            model_options = {"gpt-4o-mini": "GPT-4o (OpenAI)", "claude": "Claude (Anthropic)", "llama": "Llama (Groq)"}
            selected_model = st.selectbox("Select Model", options=list(model_options.keys()), format_func=lambda x: model_options[x], index=list(model_options.keys()).index(st.session_state.model))
            if selected_model != st.session_state.model:
                st.session_state.model = selected_model
                logger.info(f"Model changed to: {selected_model}")
            st.subheader("Research Mode")
            deep_research = st.checkbox("Enable Deep Research", help="Perform comprehensive research with more extensive web search")
            st.subheader("Document Upload")
            uploaded_file = st.file_uploader("Upload Document or Image", type=["pdf", "docx", "txt", "jpg", "jpeg", "png"], help="Upload a document for analysis or an image for visual question answering")
            if uploaded_file:
                file_path = handle_file_upload(uploaded_file)
                if file_path and file_path != st.session_state.uploaded_file_path:
                    st.session_state.uploaded_file_path = file_path
                    st.session_state.indexed = False
                    st.success(f"File '{uploaded_file.name}' uploaded successfully")
            st.subheader("Conversation Management")
            if st.button("Clear Conversation"):
                clear_conversation()
                st.rerun()
        st.title("Vaani.pro Chatbot")
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Model: {model_options[st.session_state.model]}")
        with col2:
            if st.session_state.uploaded_file_path:
                file_name = os.path.basename(st.session_state.uploaded_file_path)
                st.info(f"File: {file_name}")
            else:
                st.info("No file uploaded")
        with col3:
            if st.session_state.indexed:
                st.success("Document indexed ✓")
            elif st.session_state.uploaded_file_path and any(ext in st.session_state.uploaded_file_path for ext in ['.pdf', '.docx', '.txt']):
                st.warning("Document not indexed")
        st.markdown("### Conversation")
        display_chat_messages()
        query = st.chat_input("Type your message here...")
        if query:
            success = process_query(query=query, model=st.session_state.model, file_path=st.session_state.uploaded_file_path, deep_research=deep_research)
            if success:
                st.rerun()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal application error: {e}", exc_info=True)
        st.error(f"Application error: {str(e)}")