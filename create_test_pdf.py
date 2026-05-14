# create_test_pdf.py
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

lines = [
    "What is Artificial Intelligence?",
    "",
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "It includes machine learning, deep learning, and natural language processing.",
    "",
    "What is Machine Learning?",
    "",
    "Machine learning is a subset of AI where systems learn from data automatically.",
    "Common algorithms include linear regression, decision trees, and neural networks.",
    "",
    "What is RAG?",
    "",
    "RAG stands for Retrieval Augmented Generation.",
    "It combines a retrieval system with a language model to answer questions.",
    "Instead of relying on memorized knowledge, RAG fetches relevant documents first.",
]

for line in lines:
    pdf.cell(0, 10, line, ln=True)

pdf.output("data/test_document.pdf")
print("✅ Test PDF created at data/test_document.pdf")