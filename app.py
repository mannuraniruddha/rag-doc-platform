import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables (for local .env use)
load_dotenv()

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

# Custom CSS for the "Enterprise Dark" theme
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #11162b 100%);
        color: #e8ecf8;
    }
    .block-container {
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 1.6rem;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: 0.2px;
    }
    .app-subtitle {
        color: #b8c0dd;
        margin-bottom: 1rem;
    }
    .chip {
        display: inline-block;
        border: 1px solid #2d3a67;
        color: #c7d2ff;
        background: rgba(45, 58, 103, 0.28);
        border-radius: 999px;
        padding: 0.3rem 0.65rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        font-size: 0.82rem;
    }
    .stChatMessage {
        border: 1px solid rgba(126, 141, 196, 0.22);
        border-radius: 14px;
        background: rgba(16, 22, 44, 0.74);
        backdrop-filter: blur(2px);
        padding: 0.6rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """
You are a document QA assistant.
Answer the question using ONLY the context provided below.
If the answer is not present in the context, say:
"I could not find that in the uploaded document(s)."
Keep the answer concise and factual.

Context:
{context}

Question:
{question}
"""

# --- CORE FUNCTIONS ---

def _get_google_api_key() -> str:
    """Safely retrieves key from Streamlit secrets or environment variables."""
    # 1. Try Streamlit Secrets (for Cloud deployment)
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        # If secrets file is missing locally, ignore the error and move to env
        pass
    
    # 2. Try Environment Variables (for Local development via .env)
    return os.getenv("GOOGLE_API_KEY", "")
    
def _load_uploaded_files(uploaded_files) -> List[Document]:
    """Processes uploaded files and converts them into LangChain Documents."""
    docs: List[Document] = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif suffix in {".txt", ".md"}:
                loader = TextLoader(temp_path, encoding="utf-8")
            else:
                continue
            
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = uploaded.name
            docs.extend(loaded_docs)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return docs

def _build_vectorstore(docs: List[Document], api_key: str) -> FAISS:
    """Splits documents and builds a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_documents(chunks, embeddings)

def _answer_question(vectorstore: FAISS, question: str, api_key: str) -> str:
    """Performs retrieval and generation using the LCEL chain syntax."""
    # 1. Retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 2. Generation (Gemini 1.5 Flash)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    
    # Simple LCEL Chain
    chain = prompt | model
    response = chain.invoke({"context": context, "question": question})
    return response.content

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Upload files to ground the AI in your specific data.")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=_get_google_api_key(),
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", use_container_width=True):
        if not api_key:
            st.error("Missing Google API Key.")
        elif not uploaded_files:
            st.error("No documents uploaded.")
        else:
            with st.spinner("Analyzing and Indexing..."):
                documents = _load_uploaded_files(uploaded_files)
                if documents:
                    st.session_state.vectorstore = _build_vectorstore(documents, api_key)
                    st.success(f"Indexed {len(documents)} document pages.")
                else:
                    st.error("Could not extract text from files.")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT UI ---
st.markdown('<div class="app-title">RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Powered by Gemini 1.5 & LangChain. Grounded in your data.</div>',
    unsafe_allow_html=True,
)

# Display file "chips" if files are uploaded
if uploaded_files:
    chips = "".join([f'<span class="chip">{f.name}</span>' for f in uploaded_files])
    st.markdown(chips, unsafe_allow_html=True)

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Logic
if user_query := st.chat_input("Ask a question about the documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Validate state before processing
    if not api_key:
        ans = "Please provide a Google API Key in the sidebar."
    elif st.session_state.vectorstore is None:
        ans = "Please upload and 'Process' documents before asking questions."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Consulting documents..."):
                ans = _answer_question(st.session_state.vectorstore, user_query, api_key)
                st.markdown(ans)
    
    # Save assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": ans})