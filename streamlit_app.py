"""
=============================================================
 STREAMLIT UI — Smart PDF Q&A Bot
=============================================================
 This is the frontend that users interact with.
 It communicates with the Flask backend via HTTP requests.
 
 Features:
   - PDF upload sidebar
   - Chat interface for Q&A
   - Shows agent's thinking process (which tools it used)
=============================================================
"""

import streamlit as st
import requests
import config

# ---------- Configuration ----------
FLASK_URL = config.FLASK_URL

# ---------- Page Config ----------
st.set_page_config(
    page_title="Smart PDF Q&A Bot",
    page_icon="📄",
    layout="wide",
)

# ---------- Custom CSS for better styling ----------
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        font-size: 2rem !important;
        margin-bottom: 0.3rem !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0;
    }
    
    /* Chat message styling */
    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .bot-msg {
        background: #f0f2f6;
        color: #1a1a2e;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        font-size: 0.95rem;
    }
    
    /* Thinking steps */
    .thinking-box {
        background: #1a1a2e;
        color: #a0e7a0;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        border-left: 3px solid #667eea;
    }
    
    /* Upload success */
    .upload-success {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 12px 18px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Concept tags */
    .concept-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .tag-model { background: #e3f2fd; color: #1565c0; }
    .tag-prompt { background: #f3e5f5; color: #7b1fa2; }
    .tag-chain { background: #e8f5e9; color: #2e7d32; }
    .tag-tool { background: #fff3e0; color: #e65100; }
    .tag-agent { background: #fce4ec; color: #c62828; }
    .tag-index { background: #e0f7fa; color: #00695c; }
</style>
""", unsafe_allow_html=True)


# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = True
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = config.DEFAULT_PDF


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📄 Document Q&A")
    st.markdown("Upload a PDF and ask questions dynamically.")
    
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Check if the uploaded file is different from the active one
        if uploaded_file.name != st.session_state.pdf_filename:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{FLASK_URL}/api/upload", files=files, timeout=120)
                    
                    if response.status_code == 200:
                        st.session_state.pdf_filename = uploaded_file.name
                        st.session_state.pdf_uploaded = True
                        st.session_state.messages = []  # Reset chat history for new PDF
                        st.toast(f"✅ Loaded {uploaded_file.name} successfully!")
                        st.rerun()
                    else:
                        error_msg = response.json().get("error", "Unknown error")
                        st.error(f"❌ Upload failed: {error_msg}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

    st.markdown("---")
    st.markdown(f"📎 **Active PDF:** `{st.session_state.pdf_filename}`")

    # Clear button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        try:
            requests.post(f"{FLASK_URL}/api/clear", timeout=10)
        except Exception:
            pass
        st.session_state.messages = []
        st.rerun()

    # Concept Reference
    st.markdown("---")
    st.markdown("## 🧠 Core RAG Concepts")
    st.markdown("""
    1. 🤖 **Model** — ChatGroq LLM
    2. 📝 **Prompt** — Templates
    3. 🔗 **Chain** — Pipelines
    4. 📚 **Index** — FAISS Store
    """)


# ---------- Main Area: Header ----------
st.markdown("""
<div class="main-header">
    <h1>📄 Smart PDF Q&A Bot</h1>
    <p>Upload a PDF and ask questions — powered by LangChain Agent with Tools</p>
</div>
""", unsafe_allow_html=True)

# Show concept tags
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <span class="concept-tag tag-model">🤖 Model</span>
    <span class="concept-tag tag-prompt">📝 Prompt</span>
    <span class="concept-tag tag-chain">🔗 Chain</span>
    <span class="concept-tag tag-index">📚 Index</span>
</div>
""", unsafe_allow_html=True)


# ---------- Main Area: Chat ----------
if not st.session_state.pdf_uploaded:
    st.info("👈 Upload a PDF from the sidebar to get started!")
else:
    # Display chat messages
    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Retrieved Source Chunks"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"**Source Chunk #{i} (Page {src['page']})**")
                            st.caption(src["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your PDF..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        # Get RAG response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching PDF and generating answer..."):
                try:
                    response = requests.post(
                        f"{FLASK_URL}/api/ask",
                        json={"question": question},
                        timeout=120,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.write(answer)
                        
                        if sources:
                            with st.expander("📚 Retrieved Source Chunks"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(f"**Source Chunk #{i} (Page {src['page']})**")
                                    st.caption(src["content"])

                        # Save to session
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    else:
                        error = response.json().get("error", "Unknown error")
                        st.error(f"❌ {error}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure Flask is running!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
