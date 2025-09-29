
import streamlit as st
import os
import json
import tempfile
from datetime import datetime
from backend.ollama_chatbot import OllamaPDFChatbot, RateLimiter
from backend.ocr import extract_text_from_scanned_pdf
from backend.session_manager import SessionManager
from backend.utils import format_for_json, format_for_txt
from backend.chunker import TextChunker
from backend.pdf_loader import extract_pdf_text, is_scanned_pdf, get_pdf_metadata
from backend.text_search import TextSearcher

# Set page configuration
st.set_page_config(
    page_title="Policy Navigation Using AI",
    page_icon="📄",
    layout="wide"
)

# Simple styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem;
    margin-bottom: 2rem;
    border-radius: 30px;
    color: Pink;
    background-color: #3B1DC3;
}
.status-box {
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}
.success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #ADC31E;
}
.error {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
.info {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
}
.preview-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 1rem;
    max-height: 300px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.9em;
}
.chunk-preview {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 3px;
    padding: 0.5rem;
    margin: 0.3rem 0;
    font-size: 0.8em;
}
</style>
""", unsafe_allow_html=True)



# 🔑 Initialize session state first
SessionManager.initialize_session_state()

# Now it's safe to use st.session_state

model = st.session_state.get("selected_model", "gemma3:1b")

# Initialize session state
def initialize_session_state():
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "pdf_chunks" not in st.session_state:
        st.session_state.pdf_chunks = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama2"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "show_preview" not in st.session_state:
        st.session_state.show_preview = False
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = {}

initialize_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>📄 Policy Navigation Using AI</h1>
    <p>Upload → Extract → PreviewChat with your PDF content</p>
</div>
""", unsafe_allow_html=True)

# Check Ollama connection
pdf_chatbot = OllamaPDFChatbot()
connection_status = pdf_chatbot.check_connection()

if connection_status:
    st.markdown('<div class="status-box success">✅ AI is ready to chat!</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-box error">❌ AI is not available. Please start Ollama first.</div>', unsafe_allow_html=True)

st.markdown("---")

# Step 1: Upload PDF
st.subheader("📁 Upload Your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"📄 **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
    with col2:
        if st.button("🚀 Extract Text", type="primary"):
            st.session_state.processing = True
            st.rerun()
    with col3:
        if st.button("🗑️ Clear"):
            for key in ["pdf_text", "pdf_chunks", "chat_history", "show_preview", "pdf_metadata"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Step 2: Process PDF and Show Preview
if uploaded_file and st.session_state.processing:
    with st.spinner("🔄 Extracting text from PDF..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            # Get metadata
            metadata = get_pdf_metadata(pdf_path)
            st.session_state.pdf_metadata = metadata

            # Extract text
            text = extract_pdf_text(pdf_path)

            # Try OCR if needed
            if not text.strip() or is_scanned_pdf(pdf_path):
                st.info("📖 Running OCR on scanned document...")
                success, ocr_text, _ = extract_text_from_scanned_pdf(pdf_path)
                if success:
                    text = ocr_text

            # Save extracted text
            st.session_state.pdf_text = text
            st.session_state.processing = False
            st.session_state.show_preview = True

            # Clean up
            os.remove(pdf_path)

            st.success(f"✅ Text extracted successfully! ({len(text):,} characters)")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.processing = False
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.remove(pdf_path)

# Step 3: Preview Extracted Data
if st.session_state.show_preview and st.session_state.pdf_text:
    st.markdown("---")
    st.subheader("👀Preview Extracted Data")
    
    # Show metadata
    if st.session_state.pdf_metadata:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Pages", st.session_state.pdf_metadata.get("page_count", "N/A"))
        with col2:
            st.metric("📝 Characters", f"{len(st.session_state.pdf_text):,}")
        with col3:
            st.metric("📊 Words", f"{len(st.session_state.pdf_text.split()):,}")

    # Preview text
    st.write("**📖 Text Preview:**")
    preview_text = st.session_state.pdf_text[:2000] + ("..." if len(st.session_state.pdf_text) > 2000 else "")
    st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)
    
    # Download options for extracted text
    st.write("**💾 Download Extracted Text:**")
    col1, col2 = st.columns(2)
    
    with col1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "📄 Download as TXT",
            data=st.session_state.pdf_text,
            file_name=f"extracted_text_{timestamp}.txt",
            mime="text/plain"
        )
    
    with col2:
        json_data = {
            "metadata": st.session_state.pdf_metadata,
            "extracted_text": st.session_state.pdf_text,
            "extraction_timestamp": datetime.now().isoformat(),
            "character_count": len(st.session_state.pdf_text)
        }
        st.download_button(
            "📋 Download as JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"extracted_data_{timestamp}.json",
            mime="application/json"
        )

    # Button to proceed to chunking
    if st.button("➡️ Proceed to Chunking & Chat", type="primary"):
        # Create chunks
        with st.spinner("🔄 Creating text chunks for AI processing..."):
            chunker = TextChunker(chunk_size=1000, overlap=200)
            chunks = chunker.chunk_text(st.session_state.pdf_text)
            st.session_state.pdf_chunks = chunks
        
        st.success(f"✅ Created {len(chunks)} text chunks for optimal AI processing!")
        st.rerun()

# Step 4: Show Chunking Preview
if st.session_state.pdf_chunks and st.session_state.show_preview:
    st.markdown("---")
    st.subheader("🔍Text Chunking Preview")
    
    st.write(f"**📊 Chunking Summary:** {len(st.session_state.pdf_chunks)} chunks created")
    
    # Show chunk preview
    with st.expander("👁️ View Text Chunks", expanded=False):
        for i, chunk in enumerate(st.session_state.pdf_chunks[:5]):  # Show first 5 chunks
            preview = chunk[:200] + ("..." if len(chunk) > 200 else "")
            st.markdown(f'<div class="chunk-preview"><strong>Chunk {i+1}:</strong> {preview}</div>', 
                       unsafe_allow_html=True)
        
        if len(st.session_state.pdf_chunks) > 5:
            st.info(f"Showing first 5 chunks. Total: {len(st.session_state.pdf_chunks)} chunks")

# Step 5: Chat Interface
if st.session_state.pdf_chunks:
    st.markdown("---")
    st.subheader("💬Chat with Document")

    # AI Settings
    with st.expander("AI Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if connection_status:
                available_models = pdf_chatbot.get_available_models()
                if available_models:
                    st.session_state.selected_model = st.selectbox(
                        "🤖 AI Model:", available_models, 
                        index=0 if st.session_state.selected_model not in available_models else available_models.index(st.session_state.selected_model)
                    )
                else:
                    st.warning("No Ollama models found. Install one with: `ollama pull llama2`")
        with col2:
            st.session_state.temperature = st.slider(
                "🌡️ Creativity (Temperature):", 
                min_value=0.1, max_value=1.0, 
                value=st.session_state.temperature, step=0.1
            )

    # Chat input
    user_question = st.text_area(
        "💭 Ask :",
        placeholder="What is this document about? What are the main topics? Can you summarize the key points?",
        height=100
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Ask AI", type="primary", disabled=not user_question.strip() or not connection_status):
            with st.spinner("🤖 AI is analyzing..."):
                try:
                    # Get relevant text chunks
                    searcher = TextSearcher(st.session_state.pdf_chunks)
                    context_chunks = searcher.search_relevant_chunks(user_question, 5)
                    context = " ".join(context_chunks)

                    # Create system prompt
                    system_prompt = f"""You are a helpful AI assistant analyzing a PDF document. 
                    The document has {len(st.session_state.pdf_chunks)} sections.
                    Answer based on the provided document content. Be specific and reference relevant parts.
                    If information isn't in the document, say so clearly."""

                    # Get AI response
                    response = pdf_chatbot.get_response(
                        user_question,
                        context=context,
                        system_prompt=system_prompt
                    )

                    # Add to history
                    st.session_state.chat_history.append({
                        "user": user_question,
                        "bot": response,
                        "timestamp": datetime.now().isoformat(),
                        "chunks_used": len(context_chunks)
                    })
                    

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# Chat history display (full width)
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")
    
    # Display chat messages using safe Streamlit chat components
    for chat in reversed(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat['user'])
        
        # Bot message
        with st.chat_message("assistant"):
            st.write(chat['bot'])



# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    🤖 Powered by Ollama • 🔍 OCR with Tesseract • 📄 Built with Streamlit
</div>
""", unsafe_allow_html=True)
