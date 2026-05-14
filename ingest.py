from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings          # ✅ updated
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
    print(f"✅ Loaded {len(documents)} page(s) from your documents")
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

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model loaded")
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
        # ✅ Clear old vectors before re-uploading
        print("🗑️  Clearing old vectors from Pinecone...")
        pc.Index(index_name).delete(delete_all=True)
        print("✅ Old vectors cleared")

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
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    upload_to_pinecone(chunks, embeddings)
    print("\n🎉 Ingestion complete! Your documents are in Pinecone.")

if __name__ == "__main__":
    main()