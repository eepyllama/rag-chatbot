# chain.py
from dotenv import load_dotenv
from llm import get_llm
from retriever import get_vectorstore, get_retriever, get_prompt_template, build_rag_chain

load_dotenv()

def get_chain():
    vectorstore = get_vectorstore()
    retriever   = get_retriever(vectorstore)
    prompt      = get_prompt_template()
    llm         = get_llm()
    chain       = build_rag_chain(llm, retriever, prompt)
    return chain


if __name__ == "__main__":
    print("\n🦙 RAG Chatbot ready! Ask me anything about your documents.\n")

    # ✅ Build chain ONCE before the loop
    chain = get_chain()
    print("\n💬 Chat started! Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\n🤖 Thinking...\n")

        # ✅ Reuse the same chain every question
        answer = chain.invoke(question)
        print(f"Bot: {answer}\n")
        print("-" * 60)