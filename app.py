# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from chain import get_chain_with_history, ask_with_history
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

try:
    for key, val in st.secrets.items():
        os.environ.setdefault(key, str(val))
except Exception:
    pass

# ── Embedding model (same as ingest.py) ──────────────────
@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ── Ingest uploaded PDF into Pinecone ────────────────────
def ingest_pdf(uploaded_file):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load + split
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Upload to Pinecone
    embeddings = get_embeddings()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "rag-chatbot"

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Clear old vectors safely
    try:
        pc.Index(index_name).delete(delete_all=True)
    except Exception:
        pass

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    os.unlink(tmp_path)  # clean up temp file
    return len(chunks)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🦙",
    layout="centered"
)

st.title("🦙 RAG Chatbot")
st.caption("Powered by LangChain · Pinecone · Llama 3.1")

# ── Sidebar — PDF Upload ──────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF to chat with",
        type=["pdf"]
    )

    if uploaded_file:
        if st.button("⚡ Process PDF"):
            with st.spinner("📄 Reading and indexing your PDF..."):
                num_chunks = ingest_pdf(uploaded_file)

            st.success(f"✅ Indexed {num_chunks} chunks!")

            # Reset chat when new doc is uploaded
            st.session_state.chat_history = []
            st.session_state.messages     = []
            st.session_state.chain        = get_chain_with_history()
            st.rerun()

    st.divider()
    st.header("🛠️ Tech Stack")
    st.markdown("""
- 🔗 **LangChain** — RAG pipeline
- 🌲 **Pinecone** — vector database
- 🦙 **Llama 3.1** via Groq — LLM
- 🤗 **HuggingFace** — embeddings
- 🎈 **Streamlit** — UI
    """)

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.messages     = []
        st.rerun()

# ── Session state ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("🔗 Loading RAG chain..."):
        st.session_state.chain = get_chain_with_history()

# ── Welcome message if no chat yet ───────────────────────
if not st.session_state.messages:
    st.info("👈 Upload a PDF in the sidebar to get started, or ask about pre-loaded documents!")

# ── Display messages ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────
if question := st.chat_input("Ask me anything about your documents..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            answer = ask_with_history(
                st.session_state.chain,
                question,
                st.session_state.chat_history
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))