from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings():
    """Create the Gemini embedding model used for semantic search."""
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )