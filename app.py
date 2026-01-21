"""
Streamlit Web UI for RAG Chatbot
A user-friendly interface for interacting with the chatbot and managing documents.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import (
    ingest_file,
    ingest_urls,
    ingest_text,
    load_vector_store,
    clear_vector_store,
    CHROMA_PERSIST_DIRECTORY
)
from src.rag_chain import RAGChatbot
from src.job_scraper import get_copenhagen_it_jobs, LinkedInJobScraper

# Page config
st.set_page_config(
    page_title="Copenhagen IT Job Search - RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChat message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sidebar-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store_ready" not in st.session_state:
        # Check if vector store exists
        st.session_state.vector_store_ready = os.path.exists(CHROMA_PERSIST_DIRECTORY)


def initialize_chatbot():
    """Initialize or reinitialize the chatbot."""
    try:
        vector_store = load_vector_store()
        if vector_store:
            st.session_state.chatbot = RAGChatbot(vector_store=vector_store)
            st.session_state.vector_store_ready = True
            return True
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
    return False


def render_sidebar():
    """Render the sidebar with data management options."""
    with st.sidebar:
        st.markdown("## 📚 Knowledge Base")
        
        # Status indicator
        if st.session_state.vector_store_ready:
            st.success("✅ Knowledge base loaded")
        else:
            st.warning("⚠️ No knowledge base found")
        
        st.markdown("---")
        
        # Data ingestion options
        st.markdown("### Add Data")
        
        data_source = st.radio(
            "Select data source:",
            ["💼 Load IT Jobs", "📝 Text Input", "🔗 URL", "📄 File Upload"],
            label_visibility="collapsed"
        )
        
        if data_source == "💼 Load IT Jobs":
            render_job_loader()
        elif data_source == "📝 Text Input":
            render_text_input()
        elif data_source == "🔗 URL":
            render_url_input()
        else:
            render_file_upload()
        
        st.markdown("---")
        
        # Management options
        st.markdown("### ⚙️ Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload", use_container_width=True):
                if initialize_chatbot():
                    st.success("Reloaded!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear KB", use_container_width=True):
                clear_vector_store()
                st.session_state.vector_store_ready = False
                st.session_state.chatbot = None
                st.session_state.messages = []
                st.success("Cleared!")
                st.rerun()
        
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_history()
            st.rerun()


def render_job_loader():
    """Render job search loader section."""
    st.markdown("#### 🇩🇰 Copenhagen IT Jobs")
    st.markdown("Load current IT & Software job listings from Copenhagen.")
    
    # Show what will be loaded
    with st.expander("ℹ️ What's included?"):
        st.markdown("""
        **Companies featured:**
        - Novo Nordisk, Maersk, Spotify
        - LEGO, Microsoft, Danske Bank
        - Pleo, Unity, Trustpilot, and more
        
        **Job types:**
        - Software Engineers
        - Full Stack Developers
        - Data Engineers
        - DevOps/SRE
        - ML Engineers
        - IT Security
        
        **Information provided:**
        - Job descriptions
        - Salary ranges (DKK)
        - Required skills
        - Company benefits
        """)
    
    if st.button("🔄 Load Copenhagen IT Jobs", use_container_width=True):
        with st.spinner("Fetching job listings..."):
            try:
                jobs_text = get_copenhagen_it_jobs()
                ingest_text(jobs_text, "copenhagen_it_jobs")
                initialize_chatbot()
                st.success("✅ Loaded 12 IT job listings from Copenhagen!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading jobs: {e}")


def render_text_input():
    """Render text input section."""
    with st.form("text_form"):
        source_name = st.text_input("Source name:", placeholder="e.g., Company FAQ")
        text_content = st.text_area(
            "Enter your text:",
            height=150,
            placeholder="Paste your content here..."
        )
        
        if st.form_submit_button("➕ Add Text", use_container_width=True):
            if text_content.strip():
                with st.spinner("Processing..."):
                    try:
                        ingest_text(text_content, source_name or "user_input")
                        initialize_chatbot()
                        st.success("✅ Text added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text.")


def render_url_input():
    """Render URL input section."""
    with st.form("url_form"):
        urls_input = st.text_area(
            "Enter URLs (one per line):",
            height=100,
            placeholder="https://example.com/page1\nhttps://example.com/page2"
        )
        
        if st.form_submit_button("➕ Add URLs", use_container_width=True):
            urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
            if urls:
                with st.spinner("Fetching and processing URLs..."):
                    try:
                        ingest_urls(urls)
                        initialize_chatbot()
                        st.success(f"✅ {len(urls)} URL(s) added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter at least one URL.")


def render_file_upload():
    """Render file upload section."""
    uploaded_file = st.file_uploader(
        "Upload a file:",
        type=["txt", "pdf", "md"],
        help="Supported formats: TXT, PDF, MD"
    )
    
    if uploaded_file:
        if st.button("➕ Process File", use_container_width=True):
            with st.spinner("Processing file..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"./temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Ingest the file
                    ingest_file(temp_path)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    initialize_chatbot()
                    st.success(f"✅ File '{uploaded_file.name}' added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


def render_chat_interface():
    """Render the main chat interface."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💼 Copenhagen IT Job Search Assistant</h1>
        <p>Find your next Software & IT job in Copenhagen</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if chatbot is ready
    if not st.session_state.vector_store_ready:
        st.info("👈 Click **'Load Copenhagen IT Jobs'** in the sidebar to get started!")
        
        # Show example
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
            ### How to use this Job Search Assistant:
            
            1. **Load job data** using the sidebar:
               - Click "Load Copenhagen IT Jobs" to get current listings
               - Or add your own job descriptions
            
            2. **Ask questions** like:
               - "What Python developer jobs are available?"
               - "Which companies offer the highest salaries?"
               - "Find me junior positions with good benefits"
               - "What skills are most in demand?"
               - "Show me remote-friendly jobs"
            
            3. **The assistant will:**
               - Search through all job listings
               - Find relevant opportunities
               - Provide detailed information
            
            ### Featured Companies:
            🏢 Novo Nordisk, Maersk, Spotify, LEGO, Microsoft, Danske Bank, Pleo, Unity, and more!
            """)
        return
    
    # Initialize chatbot if needed
    if st.session_state.chatbot is None:
        initialize_chatbot()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['source']}**")
                        st.markdown(f"_{source['content']}_")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about IT jobs in Copenhagen... (e.g., 'What Python jobs are available?')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chatbot.chat(prompt, include_sources=True)
                    response = result["answer"]
                    sources = result.get("sources", [])
                    
                    st.markdown(response)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.markdown(f"**{source['source']}**")
                                st.markdown(f"_{source['content']}_")
                                st.markdown("---")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
