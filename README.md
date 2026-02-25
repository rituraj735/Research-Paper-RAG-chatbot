# 📄 Research Paper RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that enables grounded Q&A over research papers using semantic search and LLM-based reasoning.

## 🚀 Motivation

While exploring Retrieval-Augmented Generation (RAG), LangChain, and Model Context Protocol (MCP), I realized that conceptual understanding alone wasn’t enough.

**To deeply understand:** 

1. How chunking affects retrieval

2. How embeddings behave in vector space

3. How hallucinations occur in LLM systems

4. How retrieval constrains generation

### I implemented a full end-to-end RAG pipeline from scratch.

## 🧠 What This Project Does

Given one or more research paper PDFs, the system:

1. Loads and parses PDFs

2. Splits them into overlapping semantic chunks

3. Embeds each chunk using OpenAI embeddings

4. Stores embeddings in a persistent Chroma vector database

5. Retrieves top-k relevant chunks for a user query

6. Generates a context-constrained answer using an LLM

7. Returns answer + source references

This ensures answers are grounded strictly in the indexed document content.

🏗 Architecture
```
PDF → Chunking → Embedding → Vector Store (Chroma)
                                 ↓
User Query → Query Embedding → Similarity Search (Top-k)
                                 ↓
Retrieved Context → Prompt Template → LLM → Response + Sources
```

Separation of concerns:

- Ingestion layer (**create_database.py**)

- Retrieval & generation layer (**query_data.py**)

- Embedding evaluation (**compare_embeddings.py**)

## 🛠 Tech Stack

- Python

- LangChain

 - OpenAI Embeddings

- ChromaDB (vector store)

 - RecursiveCharacterTextSplitter

## GPT-based LLM generation

### ⚙️ Setup & Installation
1️⃣ Create virtual environment (recommended: Python 3.11)
```
python3.11 -m venv .venv
source .venv/bin/activate
```
2️⃣ Install dependencies
```
pip install -r requirements.txt
```
3️⃣ Add API key

Create a .env file:
```
KEY=your_key_here
```
📦 Index Research Papers

Place PDFs inside:

data/papers/

Then run:
```
python create_database.py
```
This:

1. Splits documents

2. Generates embeddings

3. Saves them into chroma/

🔍 Query the System
```
python query_data.py "What is the main contribution of this paper?"
```
Output includes:

1. Model response

2. Source references

🧪 Embedding Evaluation

To explore embedding similarity:
```
python compare_embeddings.py
```
This evaluates:

1. Vector dimensionality

2. Pairwise semantic distance

3. Embedding similarity behavior
