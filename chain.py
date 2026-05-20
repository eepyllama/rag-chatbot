# chain.py
from dotenv import load_dotenv
import os
from llm import get_llm
from retriever import get_vectorstore, get_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

try:
    import streamlit as st
    for key, val in st.secrets.items():
        os.environ.setdefault(key, str(val))
except Exception:
    pass

# ── Prompt 1: condense follow-up into standalone question ──
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given the conversation history and a follow-up question, 
rewrite the follow-up question to be a fully standalone question that 
includes all necessary context from the history.
If it's already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Follow-up question: {question}\n\nStandalone question:"),
])

# ── Prompt 2: answer using retrieved context + history ─────
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up answers. Do not repeat the question.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain_with_history():
    vectorstore = get_vectorstore()
    retriever   = get_retriever(vectorstore)
    llm         = get_llm()

    # Step 1 — condense the question using history
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    # Step 2 — retrieve + answer
    def get_context(inputs):
        # If there's history, condense first. If first question, use as-is.
        if inputs["chat_history"]:
            standalone = condense_chain.invoke({
                "question":     inputs["question"],
                "chat_history": inputs["chat_history"]
            })
        else:
            standalone = inputs["question"]

        # Use the standalone question to search Pinecone
        docs = retriever.invoke(standalone)
        return format_docs(docs)

    # Full chain
    chain = (
        {
            "context":      RunnableLambda(get_context),
            "question":     RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def ask_with_history(chain, question: str, chat_history: list) -> str:
    return chain.invoke({
        "question":     question,
        "chat_history": chat_history
    })


if __name__ == "__main__":
    print("\n🦙 RAG Chatbot with Memory! Type 'quit' to exit.\n")

    chain        = get_chain_with_history()
    chat_history = []

    print("💬 Chain ready! Start chatting.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\n🤖 Thinking...\n")

        answer = ask_with_history(chain, question, chat_history)
        print(f"Bot: {answer}\n")
        print("-" * 60)

        # ✅ Save to memory after every turn
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))