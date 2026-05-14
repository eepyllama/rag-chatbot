# RAG Chatbot 🦙

A production-style Retrieval-Augmented Generation (RAG) chatbot 
built with LangChain, Pinecone, and Llama 3.1.

## What it does
Upload any PDF documents and ask questions about them. 
The chatbot retrieves the most relevant information from 
your documents and generates accurate, grounded answers.

## Tech Stack
- **LangChain** — orchestration and RAG pipeline
- **Pinecone** — vector database for semantic search
- **Llama 3.1** via Groq — fast, free LLM inference
- **HuggingFace** — local embedding model (all-MiniLM-L6-v2)
- **Python** — core language

## Architecture
PDF Documents → Chunking → Embeddings → Pinecone
User Question → Embed → Pinecone Search → Top Chunks → Llama 3.1 → Answer

## Setup

### 1. Clone the repo
git clone https://github.com/YOURUSERNAME/rag-chatbot.git
cd rag-chatbot

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add API keys
Create a .env file:
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key

### 5. Add your PDFs
Drop PDF files into the data/ folder

### 6. Ingest documents
python ingest.py

### 7. Run the chatbot
python chain.py
