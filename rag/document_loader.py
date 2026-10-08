from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(file_path: str):
    """
    Load a PDF and split it into overlapping text chunks.

    Each chunk keeps the original PDF filename and page number
    in its metadata so the RAG system can cite its sources.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = path.name

        # PyPDFLoader uses zero-based page numbers.
        # Store a human-friendly one-based page number.
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    return chunks