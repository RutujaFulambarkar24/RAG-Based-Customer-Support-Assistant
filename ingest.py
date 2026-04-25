import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ── Step 1: Load our secret API key from .env ──
load_dotenv()

# ── Step 2: Load the PDF ──
def load_pdf(pdf_path: str):

    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages!")
    return pages


# ── Step 3: Cut into chunks ──
def split_into_chunks(pages):
   
    print("Splitting into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = 500 characters
        chunk_overlap=50,     # overlap to keep context
        separators=["\n\n", "\n", ".", " "]  # where to cut
    )
    
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks!")
    return chunks


# ── Step 4: Store in ChromaDB ──
def store_in_chromadb(chunks):
   
    print("Creating embeddings and storing in ChromaDB...")
    
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # small, fast, FREE!
    )
    
    # Store everything in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    
    print("All chunks stored in ChromaDB!")
    print("You'll see a 'chroma_db' folder appear in your project!")
    return vectorstore


# ── Main function — runs everything ──
def ingest(pdf_path: str):
    """
    This is the master function.
    Call this once with your PDF path
    and everything gets stored automatically!
    """
    pages  = load_pdf(pdf_path)
    chunks = split_into_chunks(pages)
    store_in_chromadb(chunks)
    print("\nIngestion complete! Your PDF is ready to be queried!")


if __name__ == "__main__":

    ingest("knowledge_base.pdf")