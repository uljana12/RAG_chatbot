"""
Document Ingestion Module
Handles loading, splitting, and storing documents in the vector database.
"""

import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration - use absolute path to avoid issues with working directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIRECTORY = os.path.join(_BASE_DIR, "chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_embeddings():
    """Get Ollama embeddings instance (FREE - runs locally)."""
    return OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def get_text_splitter():
    """Get text splitter for chunking documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text_file(file_path: str) -> List[Document]:
    """Load a text file and return documents."""
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def load_from_directory(directory_path: str, glob_pattern: str = "**/*.txt") -> List[Document]:
    """Load all matching files from a directory."""
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


def load_from_urls(urls: List[str]) -> List[Document]:
    """Load content from web URLs."""
    loader = WebBaseLoader(urls)
    return loader.load()


def load_from_text(text: str, metadata: Optional[dict] = None) -> List[Document]:
    """Create a document from raw text."""
    if metadata is None:
        metadata = {"source": "user_input"}
    return [Document(page_content=text, metadata=metadata)]


def process_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = get_text_splitter()
    return text_splitter.split_documents(documents)


def create_vector_store(documents: List[Document]) -> Chroma:
    """Create a new vector store from documents."""
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    return vector_store


def load_vector_store() -> Optional[Chroma]:
    """Load existing vector store if it exists."""
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        embeddings = get_embeddings()
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    return None


def add_documents_to_store(documents: List[Document], vector_store: Optional[Chroma] = None) -> Chroma:
    """Add documents to existing vector store or create new one."""
    processed_docs = process_documents(documents)
    
    if vector_store is None:
        vector_store = load_vector_store()
    
    if vector_store is None:
        # Create new vector store
        vector_store = create_vector_store(processed_docs)
    else:
        # Add to existing store
        vector_store.add_documents(processed_docs)
    
    return vector_store


def ingest_file(file_path: str) -> Chroma:
    """Ingest a single file into the vector store."""
    if file_path.endswith('.pdf'):
        documents = load_pdf(file_path)
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        documents = load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return add_documents_to_store(documents)


def ingest_urls(urls: List[str]) -> Chroma:
    """Ingest content from URLs into the vector store."""
    documents = load_from_urls(urls)
    return add_documents_to_store(documents)


def ingest_text(text: str, source_name: str = "user_input") -> Chroma:
    """Ingest raw text into the vector store."""
    documents = load_from_text(text, {"source": source_name})
    return add_documents_to_store(documents)


def clear_vector_store():
    """Delete the vector store."""
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
    # Always recreate the directory
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    print("Vector store cleared.")


if __name__ == "__main__":
    # Example usage
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and can reason.
    
    Key components include:
    1. Models - LLM wrappers for various providers
    2. Prompts - Templates for structuring inputs
    3. Chains - Combining multiple components
    4. Agents - Dynamic decision making
    5. Memory - Persisting state across interactions
    """
    
    print("Ingesting sample text...")
    vector_store = ingest_text(sample_text, "langchain_intro")
    print(f"Vector store created with {vector_store._collection.count()} documents")
