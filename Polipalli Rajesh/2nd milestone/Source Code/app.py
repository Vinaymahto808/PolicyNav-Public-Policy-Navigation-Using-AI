import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import json
import os
import tempfile

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="PDF or CSV Text Extractor & Analyzer",
    page_icon="📂",
    layout="wide"
)

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown(
    """
    <style>
    .stFileUploader {
        border: 3px dashed #ff9800;
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        color: white;
        font-weight: bold;
    }
    .stFileUploader label {
        font-size: 18px;
        font-weight: bold;
        color: #4a148c;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Functions for PDF Processing
# ---------------------------
def extract_text_from_pdf(uploaded_file, use_ocr=False):
    """
    Extract text from PDF using either direct text extraction or OCR
    """
    text_by_page = {}
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if use_ocr:
            st.info("Using OCR for text extraction...")
            try:
                images = convert_from_path(tmp_path)
                
                for page_num, image in enumerate(images, start=1):
                    text = pytesseract.image_to_string(image)
                    text_by_page[f"Page {page_num}"] = text
            except Exception as e:
                st.error(f"OCR failed: {e}")
                return None
                
        else:
            st.info("Using direct text extraction...")
            try:
                # Try pdfplumber first
                with pdfplumber.open(tmp_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text()
                        text_by_page[f"Page {page_num}"] = text if text else ""
                        
                # Check if we got meaningful text
                if not any(text.strip() for text in text_by_page.values()):
                    st.info("Direct extraction failed, trying OCR...")
                    return extract_text_from_pdf(uploaded_file, use_ocr=True)
                    
            except Exception as e:
                st.warning(f"pdfplumber failed: {e}. Trying PyPDF2...")
                try:
                    # Fall back to PyPDF2
                    with open(tmp_path, 'rb') as file:
                        pdf_reader = PdfReader(file)
                        
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text()
                            text_by_page[f"Page {page_num + 1}"] = text if text else ""
                    
                    # Check if we got meaningful text
                    if not any(text.strip() for text in text_by_page.values()):
                        st.info("PyPDF2 extraction failed, trying OCR...")
                        return extract_text_from_pdf(uploaded_file, use_ocr=True)
                        
                except Exception as e2:
                    st.warning(f"PyPDF2 also failed: {e2}. Trying OCR...")
                    return extract_text_from_pdf(uploaded_file, use_ocr=True)
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)
    
    return text_by_page

def format_text_for_json(extracted_text):
    """
    Format the extracted text in the specific format requested
    """
    formatted_text = ""
    
    for page_num, text in extracted_text.items():
        formatted_text += f"--- {page_num} ---\n{text}\n"
    
    return formatted_text

def save_to_json(data, filename="extracted_text.json"):
    """
    Save extracted text data to a JSON file
    """
    json_data = {
        "policies_text": data
    }
    
    return json.dumps(json_data, ensure_ascii=False, indent=2)

# ---------------------------
# Title
# ---------------------------
st.title("📂 PDF or CSV Text Extractor & Analyzer")

# ---------------------------
# Sidebar for Options
# ---------------------------
st.sidebar.header("Options")
use_ocr = st.sidebar.checkbox("Use OCR (for scanned PDFs)", value=False)
show_raw_text = st.sidebar.checkbox("Show Raw Extracted Text", value=True)

# ---------------------------
# File Upload Section
# ---------------------------
uploaded_file = st.file_uploader(
    "✨ Upload a PDF file ✨",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
    
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file, use_ocr=use_ocr)
    
    if extracted_text:
        # Format the text
        formatted_text = format_text_for_json(extracted_text)
        
        # Display success message
        st.markdown(f'<div class="success-box">✅ Successfully extracted text from {len(extracted_text)} pages</div>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 Formatted Text", "📊 Analysis", "💬 Chat with Document"])
        
        with tab1:
            if show_raw_text:
                st.text_area("Extracted Text:", formatted_text, height=400)
            else:
                st.info("Raw text display is disabled. Enable it in the sidebar.")
            
            # Download button for JSON
            json_data = save_to_json(formatted_text)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name="extracted_text.json",
                mime="application/json"
            )
        
        with tab2:
            st.subheader("Document Analysis")
            
            # Basic statistics
            total_pages = len(extracted_text)
            total_characters = sum(len(text) for text in extracted_text.values())
            total_words = sum(len(text.split()) for text in extracted_text.values())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pages", total_pages)
            with col2:
                st.metric("Total Words", total_words)
            with col3:
                st.metric("Total Characters", total_characters)
            
            # Page preview
            st.subheader("Page Previews")
            page_options = list(extracted_text.keys())
            selected_page = st.selectbox("Select a page to view:", page_options)
            
            if selected_page:
                page_text = extracted_text[selected_page]
                st.text_area(f"Content of {selected_page}:", page_text, height=200)
        
        with tab3:
            st.subheader("Chat with Your Document")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the document..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response based on document content
                response = generate_response(prompt, extracted_text)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        st.error("❌ Failed to extract text from the PDF. The file might be corrupted or password protected.")

else:
    st.info("👆 Please upload a PDF file to get started.")

# ---------------------------
# Helper Functions
# ---------------------------
def generate_response(prompt, extracted_text):
    """
    Generate a response based on the document content and user prompt
    """
    # Simple response logic - you can enhance this with more sophisticated NLP
    prompt_lower = prompt.lower()
    full_text = " ".join(extracted_text.values()).lower()
    
    # Check for specific keywords in the prompt
    if any(word in prompt_lower for word in ["hello", "hi", "hey"]):
        return "👋 Hello! I can help you analyze the uploaded PDF document. What would you like to know?"
    
    elif any(word in prompt_lower for word in ["summary", "summarize", "overview"]):
        # Create a simple summary (first 500 characters)
        full_content = " ".join(extracted_text.values())
        summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
        return f"📋 Here's a summary of the document:\n\n{summary}"
    
    elif any(word in prompt_lower for word in ["page", "pages"]):
        return f"The document has {len(extracted_text)} pages. You can view specific pages in the 'Analysis' tab."
    
    elif any(word in prompt_lower for word in ["healthcare", "medical"]):
        if any(term in full_text for term in ["healthcare", "medical", "health"]):
            return "The document discusses healthcare policy. You can find detailed information in the extracted text."
        else:
            return "I don't see significant healthcare content in this document."
    
    elif any(word in prompt_lower for word in ["education", "school"]):
        if any(term in full_text for term in ["education", "school", "student"]):
            return "The document contains information about education policy. Check the extracted text for details."
        else:
            return "This document doesn't appear to focus on education topics."
    
    else:
        return "I've analyzed the document. Is there something specific you'd like to know about its content? You can ask for a summary or about specific topics like healthcare, education, etc."

# ---------------------------
# Installation Instructions (hidden by default)
# ---------------------------
with st.expander("Installation Requirements (for developers)"):
    st.code("""
# Required packages:
pip install streamlit PyPDF2 pdfplumber pdf2image pytesseract pillow

# For OCR, you might also need to install Tesseract:
# On Ubuntu/Debian: sudo apt install tesseract-ocr
# On macOS: brew install tesseract
# On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
""", language="bash")