# app.py
import streamlit as st
from dotenv import load_dotenv
import os
from chain import get_chain_with_history, ask_with_history
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Load Streamlit secrets
try:
    for key, val in st.secrets.items():
        os.environ.setdefault(key, str(val))
except Exception:
    pass

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🦙",
    layout="centered"
)

st.title("🦙 RAG Chatbot")
st.caption("Powered by LangChain · Pinecone · Llama 3.1")

# ── Session state ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("🔗 Loading RAG chain..."):
        st.session_state.chain = get_chain_with_history()

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

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📚 About")
    st.info(
        "This chatbot answers questions from ingested documents "
        "using Retrieval-Augmented Generation (RAG)."
    )

    st.header("🛠️ Tech Stack")
    st.markdown("""
- 🔗 **LangChain** — RAG pipeline
- 🌲 **Pinecone** — vector database
- 🦙 **Llama 3.1** via Groq — LLM
- 🤗 **HuggingFace** — embeddings API
- 🎈 **Streamlit** — UI
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages     = []
        st.rerun()