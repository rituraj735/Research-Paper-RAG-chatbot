📄 Research Paper RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that enables grounded Q&A over research papers using semantic search and LLM-based reasoning.

🚀 Motivation

While exploring Retrieval-Augmented Generation (RAG), LangChain, and Model Context Protocol (MCP), I realized that conceptual understanding alone wasn’t enough.

To deeply understand:

How chunking affects retrieval

How embeddings behave in vector space

How hallucinations occur in LLM systems

How retrieval constrains generation

I implemented a full end-to-end RAG pipeline from scratch.

🧠 What This Project Does

Given one or more research paper PDFs, the system:

Loads and parses PDFs

Splits them into overlapping semantic chunks

Embeds each chunk using OpenAI embeddings

Stores embeddings in a persistent Chroma vector database

Retrieves top-k relevant chunks for a user query

Generates a context-constrained answer using an LLM

Returns answer + source references

This ensures answers are grounded strictly in the indexed document content.

🏗 Architecture
PDF → Chunking → Embedding → Vector Store (Chroma)
                                 ↓
User Query → Query Embedding → Similarity Search (Top-k)
                                 ↓
Retrieved Context → Prompt Template → LLM → Response + Sources

Separation of concerns:

Ingestion layer (create_database.py)

Retrieval & generation layer (query_data.py)

Embedding evaluation (compare_embeddings.py)

🛠 Tech Stack

Python

LangChain

OpenAI Embeddings

ChromaDB (vector store)

RecursiveCharacterTextSplitter

GPT-based LLM generation

⚙️ Setup & Installation
1️⃣ Create virtual environment (recommended: Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Add API key

Create a .env file:

OPENAI_API_KEY=your_key_here
📦 Index Research Papers

Place PDFs inside:

data/papers/

Then run:

python create_database.py

This:

Splits documents

Generates embeddings

Saves them into chroma/

🔍 Query the System
python query_data.py "What is the main contribution of this paper?"

Output includes:

Model response

Source references

🧪 Embedding Evaluation

To explore embedding similarity:

python compare_embeddings.py

This evaluates:

Vector dimensionality

Pairwise semantic distance

Embedding similarity behavior