import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Load embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)

# Load Groq model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Streamlit UI
st.set_page_config(
    page_title="Oracle Fusion AI SQL Generator",
    layout="wide"
)

st.title("Oracle Fusion AI SQL Generator")

st.write("Generate Oracle Fusion Finance SQL using AI + RAG + Groq")

question = st.text_area(
    "Ask your Oracle Fusion SQL question",
    placeholder="Example: Show unpaid supplier invoices"
)

if st.button("Generate SQL"):

    with st.spinner("Generating SQL..."):

        # Retrieve metadata
        docs = db.similarity_search(question, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""
You are an Oracle Fusion Finance SQL expert.

Generate Oracle Fusion BIP-compatible SQL.

Rules:
1. Use Oracle SQL syntax
2. Use only Fusion tables
3. Use correct joins
4. Do not invent columns

User Question:
{question}

Relevant Metadata:
{context}
"""

        # Generate SQL
        response = llm.invoke(prompt)

        st.subheader("Generated SQL")

        st.code(response.content, language="sql")