import json
from datetime import datetime
import streamlit as st

def format_for_json(extracted_text):
    return {
        "metadata": {
            "total_pages": len(extracted_text),
            "pages_with_text": sum(1 for t in extracted_text.values() if "No text" not in t),
            "extraction_time": datetime.now().isoformat(),
            "total_characters": sum(len(t) for t in extracted_text.values())
        },
        "content": extracted_text
    }

def format_for_txt(messages):
    """Format chat messages for text file export."""
    formatted_text = "PolicyNav PDF Chat History\n"
    formatted_text += "=" * 50 + "\n\n"
    
    for i, msg in enumerate(messages, 1):
        formatted_text += f"Message {i}:\n"
        formatted_text += f"User: {msg.get('user', 'N/A')}\n"
        formatted_text += f"Bot: {msg.get('bot', 'N/A')}\n"
        formatted_text += "-" * 30 + "\n\n"
    
    return formatted_text

def save_chat_history(formatted_text, filename="chat_history.txt"):
    """Save formatted chat history to a text file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        return True
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")
        return False
