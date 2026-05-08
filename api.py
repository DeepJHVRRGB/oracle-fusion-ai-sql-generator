from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class SQLRequest(BaseModel):
    question: str

# Load embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Vector DB
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)

# Load Groq model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

@app.get("/")
def home():
    return {"message": "Oracle Fusion AI SQL Generator API Running"}

@app.post("/generate-sql")
def generate_sql(request: SQLRequest):

    # Retrieve metadata
    docs = db.similarity_search(request.question, k=3)

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
{request.question}

Relevant Metadata:
{context}
"""

    # Generate SQL
    response = llm.invoke(prompt)

    return {
        "question": request.question,
        "sql": response.content
    }