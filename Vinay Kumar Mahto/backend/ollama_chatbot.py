from datetime import datetime
import requests
import json
import time
from collections import deque
from typing import List, Generator
import streamlit as st

# -------------------------------
# Config
# -------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CONTEXT_LENGTH = 4000
RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW = 120  # seconds


def initialize_session_state():
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
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -------------------------------
# Rate Limiter
# -------------------------------
class RateLimiter:
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.requests = deque()
        self.max_requests = max_requests
        self.window = window

    def allow_request(self) -> bool:
        now = time.time()
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


# -------------------------------
# Ollama PDF Chatbot
# -------------------------------
class OllamaPDFChatbot:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    # Health check
    def check_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    # Get available models
    def get_available_models(self) -> List[str]:
        endpoints = ["/api/tags", "/api/models", "/api/list"]
        for ep in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{ep}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_models(data)
            except Exception:
                continue
        return []

    def _parse_models(self, data) -> List[str]:
        if isinstance(data, list):
            return [str(m) for m in data if m]
        if isinstance(data, dict):
            if "models" in data:
                models = data["models"]
                return [m.get("name") or m.get("model") for m in models if isinstance(m, dict)]
            return list(data.keys())
        return []

    # -------------------------------
    # Streaming response with retry
    # -------------------------------
    def stream_response(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str = "",
        retries: int = 3,
        wait: int = 10
    ) -> Generator[str, None, None]:
        if not prompt.strip():
            yield "❌ Please enter a valid question."
            return

        if len(prompt) > 5000:
            yield "❌ Question too long. Keep under 5000 chars."
            return

        if not st.session_state.rate_limiter.allow_request():
            wait_time = st.session_state.rate_limiter.get_wait_time()
            yield f"⏳ Rate limit exceeded. Wait {wait_time:.1f}s."
            return

        payload = {
            "model": st.session_state.selected_model,
            "prompt": self._build_prompt(prompt, context, system_prompt),
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

        for attempt in range(retries):
            try:
                with self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=180
                ) as response:
                    if response.status_code != 200:
                        yield f"❌ Error {response.status_code}: {response.text[:500]}"
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
                            if "error" in json_line:
                                yield f"\n❌ Error: {json_line['error']}"
                                break
                        except Exception:
                            continue
                    return  # ✅ Success, exit after one attempt
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    yield f"⏳ Timeout, retrying in {wait}s... (attempt {attempt+1}/{retries})"
                    time.sleep(wait)
                else:
                    yield "⏳ Request timeout. The model may be loading or unavailable."
            except requests.exceptions.ConnectionError:
                yield "❌ Could not connect to Ollama. Run `ollama serve`."
                return
            except Exception as e:
                yield f"❌ Unexpected error: {str(e)}"
                return

    # -------------------------------
    # Non-streaming response with retry
    # -------------------------------
    def get_response(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str = "",
        retries: int = 3,
        wait: int = 10
    ) -> str:
        if not prompt.strip():
            return "❌ Please enter a valid question."

        if len(prompt) > 5000:
            return "❌ Question too long. Keep under 5000 chars."

        if not st.session_state.rate_limiter.allow_request():
            wait_time = st.session_state.rate_limiter.get_wait_time()
            return f"⏳ Rate limit exceeded. Wait {wait_time:.1f}s."

        payload = {
            "model": st.session_state.selected_model,
            "prompt": self._build_prompt(prompt, context, system_prompt),
            "stream": False,
            "options": {
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "top_k": st.session_state.top_k,
                "num_predict": st.session_state.max_tokens,
            }
        }

        for attempt in range(retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "No response generated.")
                else:
                    return f"❌ Error {response.status_code}: Could not get response from Ollama."
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
                return "⏳ Request timeout. The model may be loading or unavailable."
            except requests.exceptions.ConnectionError:
                return "❌ Could not connect to Ollama. Please make sure `ollama serve` is running."
            except Exception as e:
                return f"❌ Unexpected error: {str(e)}"

    # -------------------------------
    # Prompt builder
    # -------------------------------
    def _build_prompt(self, prompt: str, context: str, system_prompt: str) -> str:
        base_system = (
            "You are a helpful AI assistant that answers based on document content. "
            "Be precise, factual, and cite page numbers if available."
        )
        parts = []
        if context:
            parts.append(f"DOCUMENT CONTEXT:\n{context[:10000]}")
        if system_prompt:
            parts.append(f"SYSTEM PROMPT:\n{system_prompt}")
        parts.append(f"QUESTION:\n{prompt}")
        parts.append(f"INSTRUCTIONS:\n{base_system}")
        return "\n\n".join(parts)
