# retriever.py
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

# Load Streamlit secrets if running on Streamlit Cloud
try:
    import streamlit as st
    for key, val in st.secrets.items():
        os.environ.setdefault(key, str(val))
except Exception:
    pass


from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        index_name="rag-chatbot",
        embedding=embeddings
    )
    print("✅ Connected to Pinecone index")
    return vectorstore


def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("✅ Retriever ready (top-5 chunks)")
    return retriever