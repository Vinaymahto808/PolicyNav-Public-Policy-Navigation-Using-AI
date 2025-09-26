# Importing required modules:
import os
from io import StringIO
import pandas as pd
import streamlit as st
import base64
import pdfplumber
import openpyxl
import json
import ollama

# Page settings:
st.set_page_config(page_title="AI Policy OCR Extractor", page_icon="📄", layout="wide")
st.title("📑 Public Policy Document Extractor with AI")
st.image("banner.png", use_container_width=True)

st.title("📂 UPLOAD HERE!")
st.write("Upload files from your Computer")

# Sidebar controls:
st.sidebar.header("Settings")
multi = st.sidebar.toggle("Allow multiple files", value=True)
allowed_types = st.sidebar.multiselect(
    "Allowed file extensions",
    options=["csv", "xlsx", "txt", "pdf", "png", "jpg", "jpeg", "pptx"],
    default=["csv", "xlsx", "txt", "pdf", "png", "jpg", "jpeg"],
    help="Choose the type for upload."
)

save_to_disk = st.sidebar.toggle("Save uploads to a folder", value=False)
save_dir = st.sidebar.text_input("Save folder (relative path)", value="uploads")
if save_to_disk:
    os.makedirs(save_dir, exist_ok=True)
    st.sidebar.success(f'Files will be saved to: "{save_dir}"')
st.divider()

# Tabs & Metrics:
tab1, tab2 = st.tabs(["📁 Upload", "📊 Info"])

with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric("Allowed Types", "7")
    col2.metric("Files Uploaded", "0")  # we can update this dynamically
    col3.metric("Save-to-Folder", "On" if save_to_disk else "Off")
    
# Adding background img: 
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# call this at the start of your app
set_bg("background.jpg")

# Uploader:
uploaded = st.file_uploader(
    "Choose file(s) from your device",
    type=allowed_types if allowed_types else None,
    accept_multiple_files=multi,
    help="Click to select or drag-and-drop here."
)

def size(num_bytes: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"

def _safe_name(name: str) -> str:
    # Basic sanitization to avoid path traversal; keep it simple
    return os.path.basename(name).replace("..", "_").replace("/", "_").replace("\\", "_")

def _preview_file(file):
    # Common details
    st.write(f"**Name:** {file.name}")
    st.write(f"**Type:** {file.type or 'unknown'}")
    st.write(f"**Size:** {size(len(file.getvalue()))}")
    # Show previews by type
    ext = os.path.splitext(file.name.lower())[1].lstrip(".")
    # Images
    if ext in {"png","jpg","jpeg"}:
        st.image(file, caption=file.name, use_container_width=True)
    # Text
    elif ext == "txt":
        try:
            text = file.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            text = file.read().decode("utf-8", errors="ignore")
        st.text_area("Text preview", text, height=200)
    # CSV
    elif ext == "csv":
        try:
            df = pd.read_csv(StringIO(file.getvalue().decode("utf-8", errors="ignore")))
            st.dataframe(df.head(100), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read CSV: {e}")
    # Excel
    elif ext == "xlsx":
        try:
            df = pd.read_excel(file)
            st.dataframe(df.head(100), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read Excel: {e}. Tip: ensure 'openpyxl' is installed.")
    # PDF or others
    else:
        st.info("Preview not available for this file type. You can still download or save it.")

    # Reset pointer for potential re-use
    try:
        file.seek(0)
    except Exception:
        pass

def _save(file, folder: str):
    fname = _safe_name(file.name)
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path

if uploaded:
    # Normalize to list
    files = uploaded if isinstance(uploaded, list) else [uploaded]
    for idx, f in enumerate(files, start=1):
        with st.container(border=True):
            st.subheader(f"File {idx}")
            _preview_file(f)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download this file (from memory)",
                    data=f.getvalue(),
                    file_name=_safe_name(f.name),
                    mime=f.type or "application/octet-stream",
                    use_container_width=True
                )
            with col2:
                if save_to_disk:
                    path = _save(f, save_dir)
                    st.success(f"Saved to: `{path}`")
else:
    st.info("No files uploaded yet. Use the button above to select files.")
    
# =====================================================================================================    
    
# Chatbot Section:
with st.container():
    st.subheader("💬🖥️ Chat below 🔰")

    # Store chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box:
    user_input = st.chat_input("Ask anything...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        response = f"Ok! Got your Message: {user_input}"

        # Add response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the response
        with st.chat_message("assistant"):
            st.markdown(response)
            
# =====================================================================================================

# Function to extract text from pdf by PDFplumber
def extract_text_from_pdf_plumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n"
    return text

# Function to extract text from Excel
def extract_text_from_excel(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    text = ""
    for row in sheet.iter_rows(values_only=True):
        row_text = " ".join([str(cell) for cell in row if cell])
        text += row_text + "\n"
    return text

# Function to save text in JSON
def save_to_json(data, json_path="output.json"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return json_path

uploaded_file = st.file_uploader("Upload The Policy Document", type=["pdf", "xlsx"])

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File successfully uploaded: {uploaded_file.name}")
    
    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - overlap 
        return chunks
    
    def send_to_ollama(chunk, question=None, model="llama3.1:8b"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Analyze the chunk carefully."},
            {"role": "user", "content": chunk},
        ]
        if question:
            messages.append({"role": "user", "content": question})

        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]


    # Process based on file type
    if uploaded_file.name.endswith(".pdf"):
        extracted_text = extract_text_from_pdf_plumber(file_path)
        chunks = chunk_text(extracted_text, chunk_size=500, overlap=50)
    elif uploaded_file.name.endswith(".xlsx"):
        extracted_text = extract_text_from_excel(file_path)
        chunks = chunk_text(extracted_text, chunk_size=500, overlap=50)
    else:
        st.error("❌ Unsupported file type.")
        extracted_text = ""

    if extracted_text:
        # Save to JSON
        json_path = save_to_json({"text": extracted_text}, "extracted_txt_lib.json")

        st.subheader("📜 Extracted Text Preview")
        st.text_area("Extracted Text", extracted_text[:2000], height=300)

        st.download_button(
            label="📥 Download JSON",
            data=open(json_path, "rb").read(),
            file_name="policy_text_extracted.json",
            mime="application/json"
        )
        
# =====================================================================================================        
        
    # Display chunks:
    st.subheader("🔹Generated Chunks")
    for i, chunk in enumerate(chunks[:5]):  # show first 5 chunks
        st.write(f"**Chunk {i+1}:** {chunk[:500]}...")
    
    # Send to Ollama
    selected_chunk_index = st.selectbox("Choose a chunk to analyze:", range(1,len(chunks)))
    selected_chunk = chunks[selected_chunk_index]

    st.text_area("📌 Selected Chunk", selected_chunk, height=200)

    if st.button("Analyze through Ollama"):
        response = send_to_ollama(selected_chunk)
        st.subheader("🤖 Ollama's Response")
        st.write(response)

    # Follow-up QnA
    with st.form("followup_form"):
        user_question = st.text_input("💬 # Ask anything about this chunk:")
        submitted = st.form_submit_button("Submit")

    if submitted and user_question.strip() != "":
        followup_response = send_to_ollama(selected_chunk, question=user_question)
        st.subheader("🧠 Follow-up Answer")
        st.write(followup_response)
        
# =====================================================================================================