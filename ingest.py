import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# 1. Load PDFs
print("Loading PDFs from ./data...")
loader = PyPDFDirectoryLoader("./data")
docs = loader.load()

# 2. Chunking Logic
print("Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)

# 3. Embeddings (Using an open-source model since Groq doesn't host embeddings)
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Pinecone Setup
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384, # Dimension for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 5. Load into Pinecone
print("Uploading vectors to Pinecone...")
PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
print("Ingestion Complete! Your database is ready.")