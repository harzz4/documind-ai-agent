from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END


MODEL_NAME = "gemini-3.5-flash-lite"


class AgentState(TypedDict, total=False):
    question: str
    route: str
    documents: list[Document]
    answer: str
    sources: list[str]


def create_agent(retriever):
    """
    Create the DocuMind agentic RAG workflow.

    The agent first decides whether the user's question
    requires document retrieval. If required, it retrieves
    relevant chunks before generating the final answer.
    """

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    def route_question(state: AgentState):
        question = state["question"]

        prompt = f"""
You are the routing agent for a document research assistant.

Decide whether the user's question requires searching the
uploaded documents.

Return ONLY one word:
RETRIEVE
or
DIRECT

Use RETRIEVE when the question asks about information that
could be contained in the uploaded documents.

Use DIRECT for general questions that do not require the
uploaded documents.

User question:
{question}
"""

        response = llm.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            decision = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip().upper()
        else:
            decision = str(content).strip().upper()

        if "RETRIEVE" in decision:
            route = "retrieve"
        else:
            route = "direct"

        return {"route": route}

    def retrieve(state: AgentState):
        documents = retriever.invoke(state["question"])
        return {"documents": documents}

    def generate_answer(state: AgentState):
        question = state["question"]
        documents = state.get("documents", [])

        if documents:
            context_parts = []

            for doc in documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page_number", "Unknown")

                context_parts.append(
                    f"[Source: {source}, Page: {page}]\n"
                    f"{doc.page_content}"
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
You are DocuMind AI, a research assistant.

Answer the user's question using ONLY the provided
document context.

If the context does not contain enough information,
say that the information was not found in the uploaded
documents.

Be accurate, concise, and do not invent facts.

User question:
{question}

Document context:
{context}
"""
        else:
            prompt = f"""
You are DocuMind AI, a helpful research assistant.

Answer the following general question clearly and
accurately.

Question:
{question}
"""

        response = llm.invoke(prompt)

        sources = []

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_number", "Unknown")
            citation = f"{source} — Page {page}"

            if citation not in sources:
                sources.append(citation)

        content = response.content

        if isinstance(content, list):
            answer = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            answer = str(content).strip()

        return {
             "answer": answer,
             "sources": sources,
        }

    def route_after_decision(state: AgentState):
        return state["route"]

    workflow = StateGraph(AgentState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "route_question")

    workflow.add_conditional_edges(
        "route_question",
        route_after_decision,
        {
            "retrieve": "retrieve",
            "direct": "generate_answer",
        },
    )

    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()