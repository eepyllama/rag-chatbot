from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

try:
    import streamlit as st
    for key, val in st.secrets.items():
        os.environ.setdefault(key, str(val))
except Exception:
    pass


def get_vectorstore():
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"  # lightweight, no torch needed
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