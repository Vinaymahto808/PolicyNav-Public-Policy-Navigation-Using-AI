# ollama_client.py
import os
import logging
import ollama

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def query_ollama(prompt: str, model: str | None = None) -> str:
    """Send a prompt to Ollama and return only the assistant's reply text."""
    model = model or OLLAMA_MODEL
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable and professional assistant. "
                        "Answer in a clear, natural, ChatGPT-like style. "
                        "Do not include metadata, debug info, or system output."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        # ✅ Case 1: direct dict with 'message'
        if isinstance(response, dict) and "message" in response:
            return response["message"].get("content", "").strip()

        # ✅ Case 2: streaming or alt format with 'messages'
        if isinstance(response, dict) and "messages" in response:
            for msg in response["messages"]:
                if msg.get("role") == "assistant":
                    return msg.get("content", "").strip()

        # ✅ Case 3: if Ollama returns object with attributes
        if hasattr(response, "message"):
            return getattr(response.message, "content", "").strip()

        # ✅ Fallback: extract only the text part if metadata leaks in
        if isinstance(response, str):
            # remove "model='...'" style garbage if present
            if "message=Message" in response:
                # crude fallback parse
                start = response.find("content='")
                end = response.find("', thinking")
                if start != -1 and end != -1:
                    return response[start+9:end].strip()
            return response.strip()

        return str(response)

    except Exception as e:
        logging.exception("Error querying Ollama")
        return f"⚠️ Ollama error: {e}"
