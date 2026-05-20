# llm.py
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

# Load Streamlit secrets if running on Streamlit Cloud
try:
    import streamlit as st
    for key, val in st.secrets.items():
        os.environ.setdefault(key, str(val))
except Exception:
    pass


def get_llm():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024
    )
    print("✅ Llama 3.1 loaded via Groq")
    return llm