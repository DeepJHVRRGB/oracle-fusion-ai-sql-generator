import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load env
load_dotenv()

# Page config
st.set_page_config(
    page_title="Oracle Fusion AI SQL Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Oracle Fusion AI SQL Chatbot")

st.caption(
    "Generate Oracle Fusion Finance SQL using AI + RAG + Groq"
)

# Load embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)

# Load Groq
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"] == "assistant":
            st.code(msg["content"], language="sql")
        else:
            st.write(msg["content"])

# User question
question = st.chat_input(
    "Ask Oracle Fusion SQL question..."
)

# Generate SQL
if question:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # AI response
    with st.chat_message("assistant"):

        with st.spinner("Generating SQL..."):

            # Retrieve metadata
            docs = db.similarity_search(question, k=3)

            context = "\n\n".join([
                doc.page_content for doc in docs
            ])

            # Prompt
            prompt = f"""
You are an Oracle Fusion Finance SQL expert.

Generate Oracle Fusion BIP-compatible SQL.

Rules:
1. Use Oracle SQL syntax
2. Use only Fusion tables
3. Use proper joins
4. Do not invent columns
5. Return only SQL

User Question:
{question}

Relevant Metadata:
{context}
"""

            # Generate SQL
            response = llm.invoke(prompt)

            sql = response.content

            st.code(sql, language="sql")

            st.session_state.messages.append({
                "role": "assistant",
                "content": sql
            })