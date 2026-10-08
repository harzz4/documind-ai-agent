import streamlit as st
import tempfile
from pathlib import Path

from rag.document_loader import load_and_split_pdf
from rag.vector_store import create_vector_store, get_retriever
from agent.agent import create_agent


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocuMind AI")
st.subheader("Agentic RAG Research Assistant")

st.write(
    "Upload PDF documents and ask questions. "
    "The AI agent decides when it needs to retrieve information "
    "from your documents."
)

st.divider()

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("🔍 Process Documents", type="primary"):

        all_documents = []

        with st.spinner("Processing documents..."):

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                documents = load_and_split_pdf(temp_path)

                for document in documents:
                    document.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)

        st.session_state["documents"] = all_documents

        with st.spinner("Building vector database..."):
            vector_store = create_vector_store(all_documents)

        st.session_state["agent"] = create_agent(
            get_retriever(vector_store)
        )

        st.success(
            f"Successfully processed {len(uploaded_files)} document(s) "
            f"and created the knowledge base."
        )


if "agent" in st.session_state:

    st.divider()

    st.subheader("💬 Ask your documents")

    question = st.text_input(
        "Enter your question",
        placeholder="What is the main topic of this document?",
    )

    if st.button("Ask AI", type="primary") and question:

        with st.spinner("Thinking..."):

            result = st.session_state["agent"].invoke(
                {"question": question}
            )

        answer = result.get("answer", "No answer generated.")

        st.markdown("### 🤖 Answer")
        st.write(answer)

        sources = result.get("sources", [])

        if sources:
            st.markdown("### 📖 Sources")

            for source in sources:
                st.write(f"- {source}")