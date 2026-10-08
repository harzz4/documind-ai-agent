from langchain_core.documents import Document


def retrieve_documents(retriever, question: str) -> list[Document]:
    """
    Retrieve the most relevant document chunks for a question.
    """
    if not question or not question.strip():
        return []

    return retriever.invoke(question)