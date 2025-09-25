from __future__ import annotations
import time
import secrets
from pathlib import Path
from typing import List, Dict
import pandas as pd
import streamlit as st
import logging

from ocr import process_docx, process_pdf, detect_kind, save_chunks_to_json
from ollama_client import query_ollama

# Retrieval
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Virtual File Space", page_icon="📂", layout="wide")

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = secrets.token_hex(8)
if "files" not in st.session_state:
    st.session_state.files: List[Dict] = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embed_model" not in st.session_state:
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunk_texts = []

SESSION_ID = st.session_state.session_id
DEFAULT_UPLOAD_ROOT = Path("uploads") / SESSION_ID

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    accept_multiple = st.checkbox("Allow multiple files", value=True)
    allowed_ext = st.multiselect(
        "Allowed file types",
        options=["csv", "txt", "md", "json", "pdf", "docx", "png", "jpg", "jpeg", "xlsx", "mp3", "mp4"],
        default=["csv", "txt", "json", "pdf", "docx"],
    )
    max_mb = st.number_input("Max file size (MB)", min_value=1, max_value=2048, value=200)
    chunk_size = st.slider("Chunk size (words)", 400, 1200, 800, 100)
    persist = st.checkbox("Save uploads to disk", value=True)
    upload_root = st.text_input("Upload directory (per session)", value=str(DEFAULT_UPLOAD_ROOT))
    cleanup = st.button("Clear session & delete saved files")

if cleanup:
    st.session_state.files = []
    st.session_state.chat_history = []
    st.session_state.index = None
    st.session_state.chunk_texts = []
    st.success("Session cleared.")

st.title("📂 Virtual File Space")
st.markdown("Upload documents, chunk them, and ask questions powered by Ollama.")

# --- Uploader ---
uploads = st.file_uploader("Select file(s)", type=allowed_ext if allowed_ext else None, accept_multiple_files=accept_multiple)
if uploads:
    files = uploads if isinstance(uploads, list) else [uploads]
    for up in files:
        size_ok = up.size <= max_mb * 1024 * 1024
        ext_ok = any(up.name.lower().endswith(f".{e}") for e in allowed_ext) if allowed_ext else True
        if not size_ok or not ext_ok:
            st.error(f"{up.name} rejected — " + ("too large. " if not size_ok else "") + ("disallowed type." if not ext_ok else ""))
            continue

        kind = detect_kind(up.name, getattr(up, "type", None))
        data = up.read()
        saved_path = ""
        if persist:
            Path(upload_root).mkdir(parents=True, exist_ok=True)
            target = Path(upload_root) / up.name
            target.write_bytes(data)
            saved_path = str(target)

        meta = {
            "name": up.name,
            "mime": getattr(up, "type", ""),
            "size": up.size,
            "kind": kind,
            "saved_path": saved_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.files.append(meta)
        st.success(f"Uploaded: {up.name}")

        # Process immediately if docx/pdf
        if kind == "docx" and saved_path:
            chunks = process_docx(Path(saved_path), mode="words", chunk_size=chunk_size)
        elif kind == "pdf" and saved_path:
            chunks = process_pdf(Path(saved_path), chunk_size=chunk_size)
        else:
            chunks = []

        if chunks:
            texts = [c["text"] for c in chunks]
            embed_model = st.session_state.embed_model
            embeddings = embed_model.encode(texts)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(embeddings))
            st.session_state.index = index
            st.session_state.chunk_texts = texts
            st.info(f"Indexed {len(texts)} chunks for retrieval.")

            # Save chunks to JSON
            if saved_path:
                json_path = Path(saved_path).with_suffix(".chunks.json")
                save_chunks_to_json(chunks, json_path)
                st.success(f"Chunks saved locally as {json_path.name}")

                # Download button
                with open(json_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Chunks JSON",
                        data=f,
                        file_name=json_path.name,
                        mime="application/json"
                    )

# --- Inventory ---
st.subheader("Session Inventory")
if st.session_state.files:
    st.dataframe(pd.DataFrame(st.session_state.files), use_container_width=True)
else:
    st.write("No files uploaded yet.")

# --- Chat ---
st.subheader("Chat")

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (like ChatGPT)
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # Immediately show user message in chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("🤔 Processing your question..."):
            if st.session_state.index:
                embed_model = st.session_state.embed_model
                q_emb = embed_model.encode([prompt])
                D, I = st.session_state.index.search(np.array(q_emb), k=4)
                retrieved_chunks = [st.session_state.chunk_texts[i] for i in I[0]]
                context = "\n\n".join(retrieved_chunks)

                full_prompt = f"""
Based on the provided CONTEXT from documents, I can answer your question.

Document CONTEXT:
{context}

User's Question:
{prompt}

Answer clearly, naturally, and conversationally — like ChatGPT.
"""
                reply = query_ollama(full_prompt)
            else:
                reply = query_ollama(prompt)

        st.markdown(reply)

    # Save assistant reply in history
    st.session_state.chat_history.append({"role": "assistant", "content": reply})



# --- Footer ---
with st.expander("About this app"):
    st.write(f"Session ID: `{SESSION_ID}`")
