# app.py
import streamlit as st
import requests
import json
import os
import tempfile
from PyPDF2 import PdfReader
import pdfplumber
from typing import Generator, Dict, List, Optional
from datetime import datetime
import time
from collections import deque
import re

# ---------------------------
# Configuration
# ---------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CONTEXT_LENGTH = 4000
RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW = 60  # seconds

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2e8b57;
        margin: 1rem 0 0.5rem 0;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 5px solid;
        color: #ffffff;
        line-height: 1.6;
        white-space: pre-wrap;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #2c2c2c 0%, #3a3a3a 100%);
        border-left-color: #1f77b4;
    }
    .assistant-message {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-left-color: #2e8b57;
    }
    .file-info {
        background: linear-gradient(135deg, #2c2c2c 0%, #3a3a3a 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #ffffff;
        border-left: 4px solid #ff6b6b;
    }
    .stats-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffa726;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .success-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    .session-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .session-item:hover {
        background-color: #2c2c2c;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Rate Limiter Class
# ---------------------------
class RateLimiter:
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.requests = deque()
        self.max_requests = max_requests
        self.window = window
    
    def allow_request(self) -> bool:
        now = time.time()
        # Remove old requests outside the window
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
    
    def get_wait_time(self) -> float:
        if not self.requests or len(self.requests) < self.max_requests:
            return 0.0
        return max(0.0, self.requests[0] + self.window - time.time())

# ---------------------------
# Ollama Chatbot Class
# ---------------------------
class OllamaPDFChatbot:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = (10, 180)  # (connect, read) timeout

    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_models(self) -> List[str]:
        """Get available Ollama models with enhanced error handling"""
        endpoints = ["/api/tags", "/api/models", "/api/list"]
        
        for ep in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{ep}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_models(data)
            except Exception as e:
                continue
        
        return []

    def _parse_models(self, data) -> List[str]:
        """Parse models from different response formats"""
        if isinstance(data, list):
            return [str(m) for m in data if m]
        
        if isinstance(data, dict):
            if "models" in data:
                models = data["models"]
                if isinstance(models, list):
                    model_names = []
                    for model in models:
                        if isinstance(model, dict):
                            name = model.get("name") or model.get("model")
                            if name:
                                model_names.append(name)
                        else:
                            model_names.append(str(model))
                    return model_names
            return list(data.keys())
        
        return []

    def stream_response(self, prompt: str, context: str = "", system_prompt: str = "") -> Generator[str, None, None]:
        """Stream response from Ollama with enhanced error handling and rate limiting"""
        
        # Input validation
        if not prompt or not prompt.strip():
            yield "❌ Please enter a valid question."
            return
        
        if len(prompt) > 5000:
            yield "❌ Question too long. Please keep it under 5000 characters."
            return
        
        # Rate limiting check
        if not st.session_state.rate_limiter.allow_request():
            wait_time = st.session_state.rate_limiter.get_wait_time()
            yield f"⏳ Rate limit exceeded. Please wait {wait_time:.1f} seconds."
            return

        # Enhance prompt with context
        enhanced_prompt = self._build_enhanced_prompt(prompt, context, system_prompt)
        
        payload = {
            "model": st.session_state.selected_model,
            "prompt": enhanced_prompt,
            "stream": True,
            "options": {
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "top_k": st.session_state.top_k,
                "num_predict": st.session_state.max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=180,
            ) as response:
                
                if response.status_code != 200:
                    error_msg = response.text[:500]  # Limit error message length
                    yield f"❌ Error {response.status_code}: {error_msg}"
                    return

                full_response = ""
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    
                    try:
                        json_line = json.loads(raw_line.strip())
                        if "response" in json_line:
                            token = json_line["response"]
                            full_response += token
                            yield token
                            
                        # Check for errors in the stream
                        if "error" in json_line:
                            yield f"\n❌ Error: {json_line['error']}"
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        yield f"\n❌ Streaming error: {str(e)}"
                        break
                        
        except requests.exceptions.ConnectionError:
            yield "❌ Could not connect to Ollama. Please ensure it's running with: `ollama serve`"
        except requests.exceptions.Timeout:
            yield "⏳ Request timeout. The model might be too large or busy. Try a smaller model."
        except Exception as e:
            yield f"❌ Unexpected error: {str(e)}"

    def _build_enhanced_prompt(self, prompt: str, context: str = "", system_prompt: str = "") -> str:
        """Build enhanced prompt with context and instructions"""
        if not context:
            return prompt
        
        base_system_prompt = """You are a helpful AI assistant that answers questions based on the provided document content. 
        Follow these guidelines:
        1. Answer based STRICTLY on the document content
        2. If the information isn't in the document, clearly state this
        3. Be precise and factual
        4. Cite relevant sections or pages when possible
        5. If the question is unclear or cannot be answered with the document, explain why"""
        
        enhanced_prompt = f"""DOCUMENT CONTEXT:
{context[:10000]}  # Limit context length

QUESTION: {prompt}

INSTRUCTIONS: {base_system_prompt}
"""
        return enhanced_prompt

# ---------------------------
# PDF Handling Class
# ---------------------------
class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(uploaded_file) -> Dict[str, str]:
        """Extract text from PDF with enhanced error handling and formatting"""
        if uploaded_file.size > MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024}MB")
        
        text_by_page = {}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Try pdfplumber first (better for complex PDFs)
            text_by_page = PDFProcessor._extract_with_pdfplumber(tmp_path)
            
            # If no text found, try PyPDF2 as fallback
            if not text_by_page or all("No text" in text for text in text_by_page.values()):
                text_by_page = PDFProcessor._extract_with_pypdf2(tmp_path)
                
            # Clean and validate extracted text
            text_by_page = PDFProcessor._clean_extracted_text(text_by_page)
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return text_by_page

    @staticmethod
    def _extract_with_pdfplumber(file_path: str) -> Dict[str, str]:
        """Extract text using pdfplumber"""
        text_by_page = {}
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        # Clean and normalize text
                        text = ' '.join(text.split())
                        text_by_page[f"page_{i}"] = text
                    else:
                        text_by_page[f"page_{i}"] = f"Page {i} - No extractable text"
        except Exception as e:
            raise Exception(f"pdfplumber error: {str(e)}")
        
        return text_by_page

    @staticmethod
    def _extract_with_pypdf2(file_path: str) -> Dict[str, str]:
        """Extract text using PyPDF2 as fallback"""
        text_by_page = {}
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    text_by_page[f"page_{i}"] = text or f"Page {i} - No text"
        except Exception as e:
            raise Exception(f"PyPDF2 error: {str(e)}")
        
        return text_by_page

    @staticmethod
    def _clean_extracted_text(text_by_page: Dict[str, str]) -> Dict[str, str]:
        """Clean and validate extracted text"""
        cleaned = {}
        for page, text in text_by_page.items():
            if "No text" not in text and text.strip():
                # Remove excessive whitespace and normalize
                text = re.sub(r'\s+', ' ', text.strip())
                cleaned[page] = text
            else:
                cleaned[page] = text
        return cleaned

    @staticmethod
    def get_relevant_context(prompt: str, pdf_text: Dict[str, str], max_chars: int = MAX_CONTEXT_LENGTH) -> str:
        """Extract relevant context based on prompt keywords"""
        if not pdf_text:
            return ""
        
        prompt_keywords = set(word.lower() for word in re.findall(r'\w+', prompt) if len(word) > 3)
        
        # Score pages by relevance
        page_scores = []
        for page_num, text in pdf_text.items():
            if "No text" in text:
                continue
                
            text_lower = text.lower()
            score = sum(1 for keyword in prompt_keywords if keyword in text_lower)
            if score > 0:
                page_scores.append((page_num, text, score))
        
        # Sort by relevance and build context
        page_scores.sort(key=lambda x: x[2], reverse=True)
        
        context_parts = []
        total_chars = 0
        
        # Add most relevant pages first
        for page_num, text, score in page_scores:
            if total_chars + len(text) <= max_chars:
                context_parts.append(f"--- {page_num} (Relevance: {score}) ---\n{text}")
                total_chars += len(text)
        
        # If no relevant pages or need more context, add first pages
        if not context_parts or total_chars < max_chars // 2:
            for page_num, text in list(pdf_text.items())[:2]:
                if page_num not in [p[0] for p in page_scores[:3]]:  # Avoid duplicates
                    if total_chars + len(text) <= max_chars:
                        context_parts.append(f"--- {page_num} ---\n{text}")
                        total_chars += len(text)
        
        return "\n\n".join(context_parts) if context_parts else "\n".join(list(pdf_text.values())[:3])

# ---------------------------
# Session Management
# ---------------------------
class SessionManager:
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        defaults = {
            "messages": [],
            "pdf_text": {},
            "pdf_name": None,
            "selected_model": "llama2",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2000,
            "session_timestamp": datetime.now().isoformat(),
            "rate_limiter": RateLimiter(),
            "chat_sessions": {},
            "current_session": "default",
            "model_loaded": False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def create_chat_session(name: str = None):
        """Create a new chat session"""
        if not name:
            name = f"session_{len(st.session_state.chat_sessions) + 1}"
        
        st.session_state.chat_sessions[name] = {
            "messages": [],
            "timestamp": datetime.now().isoformat(),
            "pdf_name": st.session_state.pdf_name,
            "pdf_text": st.session_state.pdf_text.copy()
        }
        st.session_state.current_session = name
        return name

    @staticmethod
    def switch_chat_session(session_name: str):
        """Switch to a different chat session"""
        if session_name in st.session_state.chat_sessions:
            session = st.session_state.chat_sessions[session_name]
            st.session_state.current_session = session_name
            st.session_state.messages = session["messages"]
            st.session_state.pdf_name = session["pdf_name"]
            st.session_state.pdf_text = session.get("pdf_text", {})
            return True
        return False

    @staticmethod
    def save_current_session():
        """Save current session state"""
        if st.session_state.current_session in st.session_state.chat_sessions:
            st.session_state.chat_sessions[st.session_state.current_session].update({
                "messages": st.session_state.messages.copy(),
                "pdf_name": st.session_state.pdf_name,
                "pdf_text": st.session_state.pdf_text.copy()
            })

# ---------------------------
# Utility Functions
# ---------------------------
def format_for_json(extracted_text: Dict) -> Dict:
    """Format extracted text for JSON export"""
    return {
        "metadata": {
            "total_pages": len(extracted_text),
            "pages_with_text": sum(1 for t in extracted_text.values() if "No text" not in t),
            "extraction_time": datetime.now().isoformat(),
            "total_characters": sum(len(t) for t in extracted_text.values())
        },
        "content": extracted_text
    }

def save_chat_history(messages: List, pdf_data: Dict) -> Dict:
    """Prepare chat history for export"""
    return {
        "chat_session": {
            "timestamp": st.session_state.session_timestamp,
            "total_messages": len(messages),
            "messages": messages,
            "model_used": st.session_state.selected_model,
            "parameters": {
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "top_k": st.session_state.top_k
            }
        },
        "pdf_data": pdf_data
    }

def display_file_info():
    """Display file information card"""
    if st.session_state.pdf_text:
        total_pages = len(st.session_state.pdf_text)
        pages_with_text = sum(1 for t in st.session_state.pdf_text.values() if "No text" not in t)
        total_chars = sum(len(t) for t in st.session_state.pdf_text.values())
        
        st.markdown(f"""
        <div class="file-info">
            <h4>📄 Document Information</h4>
            <p><strong>File:</strong> {st.session_state.pdf_name}</p>
            <p><strong>Pages:</strong> {total_pages} ({pages_with_text} with text)</p>
            <p><strong>Total Characters:</strong> {total_chars:,}</p>
            <p><strong>Loaded:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Main Application
# ---------------------------
def main():
    # Initialize session state
    SessionManager.initialize_session_state()
    chatbot = OllamaPDFChatbot()
    pdf_processor = PDFProcessor()

    # Main header
    st.markdown('<div class="main-header">📚 Advanced PDF Chatbot with Ollama</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Connection status
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔍 Check Ollama Connection"):
                if chatbot.check_connection():
                    st.success("✅ Ollama is running!")
                    st.session_state.model_loaded = True
                else:
                    st.error("❌ Ollama not available")
                    st.session_state.model_loaded = False
        
        # Model selection
        st.subheader("🤖 Model Settings")
        
        try:
            models = chatbot.get_available_models()
            if models:
                st.session_state.selected_model = st.selectbox(
                    "Select Model", 
                    models, 
                    index=0
                )
            else:
                st.warning("No models found. Using default.")
                st.session_state.selected_model = st.text_input(
                    "Enter Model Name", 
                    "llama2"
                )
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.session_state.selected_model = "llama2"

        # Model parameters
        st.subheader("⚙️ Model Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.temperature = st.slider(
                "Temperature", 0.1, 1.0, st.session_state.temperature, 0.1,
                help="Lower = more deterministic, Higher = more creative"
            )
            st.session_state.top_k = st.slider(
                "Top K", 1, 100, st.session_state.top_k,
                help="Consider top K tokens only"
            )
        
        with col2:
            st.session_state.top_p = st.slider(
                "Top P", 0.1, 1.0, st.session_state.top_p, 0.1,
                help="Nucleus sampling parameter"
            )
            st.session_state.max_tokens = st.number_input(
                "Max Tokens", min_value=100, max_value=8000, value=st.session_state.max_tokens,
                help="Maximum response length"
            )

        # Session management
        st.subheader("💾 Session Management")
        
        if st.button("🆕 New Chat Session"):
            new_session = SessionManager.create_chat_session()
            st.success(f"Created session: {new_session}")
            st.rerun()
        
        # Display existing sessions
        if st.session_state.chat_sessions:
            st.write("**Active Sessions:**")
            for session_name in list(st.session_state.chat_sessions.keys())[-5:]:  # Show last 5
                if st.button(
                    f"💬 {session_name} ({len(st.session_state.chat_sessions[session_name]['messages'])} messages)",
                    key=f"btn_{session_name}"
                ):
                    SessionManager.switch_chat_session(session_name)
                    st.rerun()

        # Actions
        st.subheader("🛠️ Actions")
        
        if st.button("🗑️ Clear Current Chat"):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
            
        if st.button("🔄 Reset Everything"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            SessionManager.initialize_session_state()
            st.success("Complete reset done!")
            st.rerun()

        # Statistics
        if st.session_state.messages:
            st.subheader("📊 Statistics")
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            st.markdown(f"""
            <div class="stats-card">
                <p>💬 Messages: {len(st.session_state.messages)}</p>
                <p>👤 User: {user_msgs}</p>
                <p>🤖 Assistant: {assistant_msgs}</p>
            </div>
            """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="sub-header">📤 PDF Upload & Analysis</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload PDF Document", 
            type=["pdf"],
            help="Maximum file size: 50MB"
        )
        
        if uploaded_file:
            if st.session_state.pdf_name != uploaded_file.name:
                with st.spinner("🔄 Extracting text from PDF..."):
                    try:
                        extracted_text = pdf_processor.extract_text_from_pdf(uploaded_file)
                        if extracted_text:
                            st.session_state.pdf_text = extracted_text
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.messages = []
                            SessionManager.save_current_session()
                            
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                            
                            # Display file info
                            display_file_info()
                            
                            # Show page preview
                            with st.expander("📖 Page Preview (First 3 Pages)"):
                                for page_num, text in list(extracted_text.items())[:3]:
                                    st.write(f"**{page_num}:**")
                                    st.text(text[:500] + "..." if len(text) > 500 else text)
                        else:
                            st.error("❌ No text could be extracted from the PDF")
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
            else:
                display_file_info()

        # Export options
        if st.session_state.pdf_text:
            st.markdown('<div class="sub-header">💾 Export Options</div>', unsafe_allow_html=True)
            
            pdf_data = format_for_json(st.session_state.pdf_text)
            chat_data = save_chat_history(st.session_state.messages, pdf_data)
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(chat_data, indent=2, ensure_ascii=False),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    help="Complete chat history with PDF content"
                )
            with col_exp2:
                st.download_button(
                    "📄 Download Text",
                    "\n\n".join(st.session_state.pdf_text.values()),
                    file_name=f"pdf_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    help="Extracted PDF text only"
                )

    with col2:
        st.markdown('<div class="sub-header">💬 Chat Interface</div>', unsafe_allow_html=True)
        
        if not st.session_state.pdf_text:
            st.info("👆 Please upload a PDF document to start chatting")
            st.markdown("""
            <div class="warning-box">
                <strong>Tips for best results:</strong>
                <ul>
                    <li>Upload text-based PDFs for best accuracy</li>
                    <li>Scanned PDFs may not work well</li>
                    <li>Keep questions specific to the document content</li>
                    <li>Use clear, concise language</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display chat messages
            for i, msg in enumerate(st.session_state.messages):
                role_class = "user-message" if msg["role"] == "user" else "assistant-message"
                icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(
                    f'<div class="chat-message {role_class}"><strong>{icon} {msg["role"].title()}:</strong><br>{msg["content"]}</div>', 
                    unsafe_allow_html=True
                )

            # Chat input
            if prompt := st.chat_input("Ask a question about the PDF..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.markdown(
                    f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{prompt}</div>', 
                    unsafe_allow_html=True
                )

                # Get relevant context
                pdf_context = pdf_processor.get_relevant_context(
                    prompt, 
                    st.session_state.pdf_text,
                    MAX_CONTEXT_LENGTH
                )

                # Prepare for assistant response
                with st.spinner("🤖 Processing your question..."):
                    placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for token in chatbot.stream_response(
                        prompt, 
                        context=pdf_context,
                        system_prompt="Answer based strictly on the provided document content. Be precise and factual."
                    ):
                        full_response += token
                        placeholder.markdown(
                            f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong><br>{full_response}▌</div>', 
                            unsafe_allow_html=True
                        )

                    # Final message without cursor
                    placeholder.markdown(
                        f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong><br>{full_response}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Save assistant response
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    SessionManager.save_current_session()

            # Chat controls
            if st.session_state.messages:
                col_ctl1, col_ctl2 = st.columns(2)
                with col_ctl1:
                    if st.button("⏹️ Stop Generating", use_container_width=True):
                        st.warning("Stop functionality requires Streamlit update")
                with col_ctl2:
                    if st.button("📋 Copy Last Response", use_container_width=True):
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                            st.session_state.messages[-1]["content"]
                            st.success("Response copied to clipboard!")

# ---------------------------
# Run the application
# ---------------------------
if __name__ == "__main__":
    main()