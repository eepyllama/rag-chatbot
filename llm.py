# llm.py
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    """Load Llama 3 via Groq API"""

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",   # free, fast Llama 3 model on Groq
        temperature=0.2,               # low = more factual, less creative
        max_tokens=1024                # max length of the answer
    )

    print("✅ Llama 3.1 loaded via Groq")
    return llm