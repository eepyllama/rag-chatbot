# 🦙 RAG Chatbot

> Upload any PDF. Ask anything. Get grounded, accurate answers — powered by Llama 3.1.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![Llama](https://img.shields.io/badge/Llama_3.1-via_Groq-orange?style=flat-square)

---

## What is this?

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from your own documents — not from the LLM's general knowledge.

Instead of asking Llama 3.1 to answer from memory (and risk hallucinations), this system:
1. Searches your documents for the most relevant information
2. Feeds that information to the LLM as context
3. Returns a grounded, accurate answer

---

## Features

- 📄 **PDF Upload** — upload any PDF directly from the browser UI
- 💬 **Multi-turn Chat** — remembers conversation context across questions
- 🔍 **Semantic Search** — finds relevant chunks by meaning, not just keywords
- 🧠 **Query Condensing** — rewrites follow-up questions before searching
- 🗑️ **Clear Chat** — reset the conversation anytime from the sidebar

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  PyPDFLoader │────▶│ Text Splitter │────▶│  Embeddings │
│  (load PDF) │     │ (500 chars)  │     │ (HuggingFace│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │    Pinecone      │
                                        │ (Vector Database)│
                                        └────────┬─────────┘
                                                 │
User Question ──▶ Embed Query ──▶ Similarity Search
                                                 │
                                         Top 5 Chunks
                                                 │
                                                 ▼
                                    ┌────────────────────┐
                                    │  Llama 3.1 (Groq)  │
                                    │  + Chat History    │
                                    └────────┬───────────┘
                                             │
                                             ▼
                                       Grounded Answer
```

---

## Tech Stack

| Tool | Role |
|------|------|
| **LangChain** | RAG pipeline orchestration |
| **Pinecone** | Vector database for semantic search |
| **Llama 3.1** via Groq | Language model for answer generation |
| **HuggingFace** | Local embedding model (`all-MiniLM-L6-v2`) |
| **Streamlit** | Chat UI with PDF uploader |
| **PyPDF** | PDF text extraction |
| **Python 3.11** | Core language |

---

## Project Structure

```
rag-chatbot/
├── app.py           # Streamlit UI + PDF upload
├── chain.py         # RAG chain with chat memory
├── retriever.py     # Pinecone vector store + retriever
├── llm.py           # Llama 3.1 via Groq
├── ingest.py        # CLI document ingestion pipeline
├── requirements.txt # Python dependencies
└── .env             # API keys (never committed)
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/eepyllama/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

Get your keys from:
- [Pinecone](https://app.pinecone.io) — free tier available
- [Groq](https://console.groq.com) — free, very fast Llama 3.1 inference

### 5. (Optional) Pre-load documents via CLI
```bash
# Drop PDFs into the data/ folder first
python ingest.py
```

### 6. Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload a PDF from the sidebar and start chatting!

---

## How Chat Memory Works

Standard RAG loses context between questions:

```
You:  What is BERT?
Bot:  BERT is a transformer model...

You:  Who invented it?
Bot:  ❌ I don't know what "it" refers to.
```

This chatbot uses **query condensing** — before searching Pinecone, it rewrites the follow-up question using the full conversation history:

```
You:  What is BERT?
Bot:  BERT is a transformer model...

You:  Who invented it?
      ↓ condensed to: "Who invented BERT?"
Bot:  ✅ BERT was introduced by researchers at Google...
```

---

## Key Learnings

- RAG is retrieval + generation working together — not magic
- Embeddings are numbers that capture semantic meaning
- Prompt templates matter more than expected
- Chat memory requires query condensing, not just history passing
- Debugging cloud deployment is a skill in itself

---

## License

MIT — feel free to use, modify, and build on this.
