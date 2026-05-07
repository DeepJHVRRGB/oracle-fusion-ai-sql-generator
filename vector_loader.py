from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load files
loader1 = TextLoader("data/ap_tables.txt")
loader2 = TextLoader("data/ap_joins.txt")

docs1 = loader1.load()
docs2 = loader2.load()

documents = docs1 + docs2

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

# Free embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
db = Chroma.from_documents(
    texts,
    embedding,
    persist_directory="vector_db"
)

db.persist()

print("Vector DB created successfully")