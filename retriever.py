from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)

# User query
query = "supplier invoice tables"

# Search
results = db.similarity_search(query, k=3)

# Print results
for r in results:
    print(r.page_content)
    print("=" * 50)