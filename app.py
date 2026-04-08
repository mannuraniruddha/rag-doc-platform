import os
import tempfile
import time
from pathlib import Path

# --- CRITICAL: FORCE PRODUCTION ENDPOINT ---
os.environ["GOOGLE_API_VERSION"] = "v1"

import streamlit as st
from dotenv import load_dotenv

# --- DIRECT SDK FOR STABILITY ---
import google.generativeai as genai

# --- LANGCHAIN FOR RAG INFRASTRUCTURE ---
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# --- UI CONFIGURATION & CENTERING ---
st.set_page_config(page_title="RAG AI Assistant", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    /* Nuke Streamlit Headers and Banners */
    header, [data-testid="stHeader"], [data-testid="stDecoration"], footer {
        display: none !important;
    }
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #11162b 100%);
        color: #e8ecf8;
    }
    
    /* Absolute Centering Logic */
    .block-container {
        max-width: 900px;
        margin: auto;
        padding-top: 2rem;
    }
    .centered-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        width: 100%;
        margin-bottom: 2.5rem;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        background: -webkit-linear-gradient(#fff, #b8c0dd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-subtitle {
        color: #b8c0dd;
        font-size: 1.1rem;
    }
    
    /* File Chip Styling */
    .chip-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 2rem;
    }
    .chip {
        border: 1px solid #2d3a67;
        color: #c7d2ff;
        background: rgba(45, 58, 103, 0.28);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.85rem;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        border: 1px solid rgba(126, 141, 196, 0.2);
        border-radius: 12px;
        background: rgba(16, 22, 44, 0.6);
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC ---

def _get_api_key():
    key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error("Missing API Key. Check your .env file or Streamlit Secrets.")
        st.stop()
    return key

def _process_docs_batched(uploaded_files, api_key):
    docs = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name
        try:
            loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path)
            docs.extend(loader.load())
        finally:
            if os.path.exists(path): os.remove(path)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    # Using the standard embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # BATCHING LOGIC: Prevents 503 Server Disconnected errors
    batch_size = 20
    vectorstore = None
    
    progress_bar = st.progress(0, text="Embedding knowledge base...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        
        progress = min((i + batch_size) / len(chunks), 1.0)
        progress_bar.progress(progress, text=f"Embedded {min(i+batch_size, len(chunks))} of {len(chunks)} chunks...")
        time.sleep(0.3) # Anti-rate-limit jitter
    
    progress_bar.empty()
    return vectorstore

# --- APP START & STATE ---
api_key = _get_api_key()
genai.configure(api_key=api_key) # Initialize Direct SDK

if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "file_fingerprint" not in st.session_state: st.session_state.file_fingerprint = None

# --- MAIN UI HEADERS ---
st.markdown(
    """
    <div class="centered-header">
        <div class="main-title">RAG Document Assistant</div>
        <div class="main-subtitle">Auto-Indexing AI • Built on Gemini 2.5 Flash</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- SIDEBAR (Auto-Processing) ---
with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader("Drop Knowledge Here", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        current_fp = "-".join([f"{f.name}_{f.size}" for f in uploaded_files])
        if st.session_state.file_fingerprint != current_fp:
            with st.status("Auto-Indexing Documents...", expanded=True) as status:
                st.session_state.vectorstore = _process_docs_batched(uploaded_files, api_key)
                st.session_state.file_fingerprint = current_fp
                status.update(label="Ready to Chat!", state="complete", expanded=False)
                
    elif st.session_state.file_fingerprint is not None:
        # User cleared files, reset state
        st.session_state.vectorstore = None
        st.session_state.file_fingerprint = None

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- FILE CHIPS ---
if uploaded_files:
    chips = "".join([f'<span class="chip">{f.name}</span>' for f in uploaded_files])
    st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)

# --- CHAT DISPLAY & LOGIC ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Retrieve context via LangChain & FAISS
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                context_docs = retriever.invoke(prompt)
                context = "\n\n".join(d.page_content for d in context_docs)

                # 2. Generate answer via DIRECT Google SDK (Bypasses LangChain 404 bugs)
                try:
                    # Using the stable 2.5 production model
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    qa_prompt = f"Answer the question based ONLY on this context. If the answer is not in the context, say so.\n\nContext:\n{context}\n\nQuestion: {prompt}"
                    
                    response = model.generate_content(qa_prompt)
                    ans = response.text
                    
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
    else:
        st.info("Drop a document in the sidebar to begin.")