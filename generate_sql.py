from dotenv import load_dotenv
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)
print(os.getenv("GROQ_API_KEY"))
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

question = input("Ask your Oracle Fusion SQL question: ")

docs = db.similarity_search(question, k=3)

context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
You are an Oracle Fusion Finance SQL expert.

Generate Oracle Fusion BIP-compatible SQL.

Rules:
1. Use Oracle SQL syntax.
2. Use only tables and joins from the context.
3. Use clear table aliases.
4. Do not invent columns.
5. Return SQL and short explanation.

User Question:
{question}

Relevant Oracle Fusion Metadata:
{context}
"""

response = llm.invoke(prompt)

print("\nGenerated SQL:\n")
print(response.content)