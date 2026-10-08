# DocuMind AI — Agentic RAG Research Assistant

An agentic Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural-language questions. The AI agent intelligently decides whether document retrieval is required before generating a grounded answer with source references.

## 🚀 Features

- 📄 Upload and process PDF documents
- ✂️ Intelligent document chunking
- 🧠 Gemini-powered semantic embeddings
- 🔎 FAISS vector similarity search
- 🤖 Agentic question routing using LangGraph
- 💬 Gemini-powered answer generation
- 📚 Source-aware responses with document and page references
- 🛡️ Prevents unsupported answers when information is not found
- 🌐 Interactive Streamlit interface

## 🏗️ Architecture

                    User Question
                          │
                          ▼
                 ┌─────────────────┐
                 │  Agent Router   │
                 │    LangGraph    │
                 └────────┬────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
          Retrieval Needed?       Direct
                │                   │
                ▼                   ▼
        ┌──────────────┐      ┌─────────────┐
        │ FAISS Search │      │    Gemini   │
        └──────┬───────┘      └──────┬──────┘
               │                     │
               ▼                     │
        Relevant Chunks              │
               │                     │
               └──────────┬──────────┘
                          ▼
                 ┌─────────────────┐
                 │  Gemini Answer   │
                 └────────┬────────┘
                          ▼
                 Answer + Sources

## 🔄 RAG Pipeline

```text
PDF Document
     │
     ▼
PDF Loader
     │
     ▼
Text Splitting
     │
     ▼
Gemini Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Semantic Retrieval
     │
     ▼
Relevant Document Chunks
     │
     ▼
Gemini Generation
     │
     ▼
Grounded Answer + Sources
```

## 🤖 Agentic Workflow

The application uses LangGraph to implement an agentic workflow.

The agent first analyzes the user's question and decides whether external document retrieval is required.

**Route 1 — Retrieval**

For document-based questions:
```text
Question → Agent → FAISS Retriever → Relevant Chunks → Gemini → Answer
```
**Route 2 — Direct**

For questions that do not require document retrieval:
```text
Question → Agent → Gemini → Answer
```
This allows the system to dynamically choose the appropriate workflow instead of always performing retrieval.

## 🧠 Why This Is Agentic

Unlike a basic RAG chatbot that always retrieves documents, DocuMind AI uses an agentic decision step.

The agent determines:
- Whether retrieval is required
- Which workflow should be executed
- Whether relevant document context should be passed to the LLM
- Which sources were used to generate the response

This creates a conditional AI workflow rather than a fixed retrieval pipeline.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | LLM and RAG orchestration |
| LangGraph | Agentic workflow |
| Gemini | LLM and embeddings |
| FAISS | Vector similarity search |
| PyPDF | PDF document loading |
| Streamlit | Web interface |
| python-dotenv | Environment configuration |

## 📁 Project Structure

```text
documind-ai-agent/
│
├── agent/
│   ├── __init__.py
│   └── agent.py
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── retriever.py
│
├── utils/
│   ├── __init__.py
│   └── config.py
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Setup

**1. Clone the repository**
```text
git clone https://github.com/harzz4/documind-ai-agent.git
cd documind-ai-agent
```
**2. Create a virtual environment**
```text
python -m venv .venv
```
Activate it:

**Windows**
```text
.venv\Scripts\activate
```
**macOS/Linux**
```text
source .venv/bin/activate
```
**3. Install dependencies**
```text
pip install -r requirements.txt
```
**4. Configure Gemini API**

Create a .env file:
```text
GOOGLE_API_KEY=your_google_api_key
```
Never commit your .env file or API keys to GitHub.

**5. Run the application**
```text
streamlit run app.py
```
The application will open in your browser.

## 💡 Example

Upload a PDF and ask:
```text
What is Blitzkrieg?
```
DocuMind AI retrieves relevant sections from the document and generates a grounded explanation while showing the source pages used.

## 🔐 Security

API credentials are loaded through environment variables and are excluded from version control using **.gitignore**.

## 📌 Key Engineering Concepts

This project demonstrates:
- Agentic AI workflows
- Retrieval-Augmented Generation (RAG)
- Semantic search
- Vector databases
- LLM orchestration
- Conditional agent routing
- Document processing
- Source attribution
- Modular Python architecture
- Streamlit application development

## 🔮 Future Improvements

- Multi-document knowledge bases
- Conversation memory
- Hybrid keyword + vector retrieval
- Reranking retrieved chunks
- Streaming responses
- Evaluation metrics for RAG quality
- Persistent vector databases
- Authentication and deployment

## 👩‍💻 Author

Harini R A