from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.embeddings import get_embeddings


INDEX_PATH = Path("data/faiss_index")


def create_vector_store(documents):
    """Create a FAISS vector store from document chunks."""
    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def save_vector_store(vector_store):
    """Persist the FAISS index locally."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_PATH))


def load_vector_store():
    """Load the previously saved FAISS index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "No FAISS index found. Upload and process a PDF first."
        )

    embeddings = get_embeddings()

    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(vector_store, k=4):
    """Create a similarity retriever from the vector store."""
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )