import streamlit as st
from datetime import datetime
from backend.ollama_chatbot import RateLimiter
import copy


class SessionManager:
    @staticmethod
    def initialize_session_state():
        """Initialize Streamlit session_state with defaults."""
        defaults = {
            "messages": [],
            "pdf_text": {},
            "pdf_name": None,
            "selected_model": "gemma3:1b",
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

        # Initialize missing keys
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Ensure a default chat session always exists
        if "default" not in st.session_state.chat_sessions:
            SessionManager.create_chat_session("default")

    @staticmethod
    def create_chat_session(name: str = None):
        """Create a new chat session."""
        if not name:
            name = f"session_{len(st.session_state.chat_sessions) + 1}"

        st.session_state.chat_sessions[name] = {
            "messages": [],
            "timestamp": datetime.now().isoformat(),
            "pdf_name": st.session_state.pdf_name,
            "pdf_text": copy.deepcopy(st.session_state.pdf_text)
        }
        st.session_state.current_session = name
        return name

    @staticmethod
    def switch_chat_session(session_name: str):
        """Switch to a different chat session."""
        if session_name in st.session_state.chat_sessions:
            s = st.session_state.chat_sessions[session_name]
            st.session_state.current_session = session_name
            st.session_state.messages = copy.deepcopy(s["messages"])
            st.session_state.pdf_name = s["pdf_name"]
            st.session_state.pdf_text = copy.deepcopy(s.get("pdf_text", {}))
            return True
        return False

    @staticmethod
    def save_current_session():
        """Save the current state into the active session."""
        if st.session_state.current_session in st.session_state.chat_sessions:
            st.session_state.chat_sessions[st.session_state.current_session].update({
                "messages": copy.deepcopy(st.session_state.messages),
                "pdf_name": st.session_state.pdf_name,
                "pdf_text": copy.deepcopy(st.session_state.pdf_text)
            })

    @staticmethod
    def save_chat(chat_history):
        """Save chat history to the current session."""
        if not hasattr(st.session_state, "current_session"):
            st.session_state.current_session = "default"

        if st.session_state.current_session not in st.session_state.chat_sessions:
            SessionManager.create_chat_session(st.session_state.current_session)

        st.session_state.chat_sessions[st.session_state.current_session]["messages"] = copy.deepcopy(chat_history)
