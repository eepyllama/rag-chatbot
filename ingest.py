# ingest.py
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def load_documents(data_path="data/"):
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s)")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

from langchain_community.embeddings import FastEmbedEmbeddings

def get_embeddings():
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    print("✅ Embedding model loaded (FastEmbed)")
    return embeddings

def upload_to_pinecone(chunks, embeddings):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "rag-chatbot"

    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        # ✅ Try to clear, but don't crash if already empty
        try:
            pc.Index(index_name).delete(delete_all=True)
            print("🗑️  Old vectors cleared")
        except Exception:
            print("ℹ️  Index already empty, skipping clear")

    print("⏳ Uploading chunks to Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print(f"✅ Uploaded {len(chunks)} chunks to Pinecone!")
    return vector_store

def main():
    print("🚀 Starting document ingestion...\n")
    documents = load_documents()
    chunks    = split_documents(documents)
    embeddings = get_embeddings()
    upload_to_pinecone(chunks, embeddings)
    print("\n🎉 Ingestion complete!")

if __name__ == "__main__":
    main()