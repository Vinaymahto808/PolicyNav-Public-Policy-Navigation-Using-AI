import streamlit as st

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Streamlit File Uploader",
    page_icon="📂"
)

# ---------------------------
# Custom Styling for File Uploader
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
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Title
# ---------------------------
st.title("📂 Upload Your File Here")

# ---------------------------
# File Upload Section
# ---------------------------
uploaded_file = st.file_uploader(
    "✨ Choose a file to upload ✨", 
    type=["jpg", "png", "pdf", "txt"]
)

if uploaded_file is not None:
    st.success(f"✅ Successfully uploaded: {uploaded_file.name}")

# ---------------------------
# Chatbox Section
# ---------------------------
st.subheader("💬 Ask Your Questions")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user message
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate simple bot response (placeholder logic)
    bot_reply = f"🤖 I received your message: {user_input}"

    # Save and display bot response
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)



