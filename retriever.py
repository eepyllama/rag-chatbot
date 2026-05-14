from dotenv import load_dotenv
import os

load_dotenv()


from langchain_huggingface import HuggingFaceEmbeddings   # ✅ fix 2
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Key loaded:", bool(os.getenv("PINECONE_API_KEY")))


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"        # ✅ fix 1
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
        search_kwargs={"k": 3}
    )

    print("✅ Retriever ready (top-3 chunks)")
    return retriever

def get_prompt_template():
    template = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up answers.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    print("✅ Prompt template created")
    return prompt

def build_rag_chain(llm, retriever, prompt):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain built (LCEL style)")
    return rag_chain

def test_retrieval(query="What is this document about?"):
    vectorstore = get_vectorstore()
    retriever = get_retriever(vectorstore)

    print(f"\n🔍 Query: '{query}'")                                 # ✅ fix 3
    print("-" * 50)

    results = retriever.invoke(query)

    for i, doc in enumerate(results, 1):
        print(f"\n📄 Chunk {i}:")
        print(doc.page_content[:300])
        print(f"   Source: {doc.metadata.get('source', 'unknown')}")

    return results

if __name__ == "__main__":
    test_retrieval("What is this document about?")