# Agentic RAG

A  **Retrieval-Augmented Generation (RAG)** system with **Agentic Workflow** using **LangGraph**, **Streamlit**, **OpenAI**, and **Pinecone**.

This app allows you to **upload a document** (PDF or Excel), **embed** it into Pinecone, and ask **contextual questions** powered by GPT.

---

## Features

- **Document Upload**: Supports PDF and Excel files.
- **OpenAI Embeddings**: Uses `text-embedding-3-small`.
- **Similarity Search**: Retrieves top-matching chunks from Pinecone.
- **LLM-Powered Q&A**: Uses GPT to answer based on retrieved context.
- **Agentic Workflow**: Dynamically manages ingestion & query flows.
- **Streamlit UI**: Simple web interface for interaction.

---
