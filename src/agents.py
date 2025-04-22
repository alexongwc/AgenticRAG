# 📁 src/agents.py
import os
import ast
from tempfile import NamedTemporaryFile
import streamlit as st

from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone_utils import embedding_model, index_name

# Set up OpenAI client using secret
api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
print("🔑 DEBUG: OPENAI_API_KEY =", "✅ FOUND" if api_key else "❌ MISSING")
print("📦 Using Pinecone index:", index_name)

def load_and_split_docs(uploaded_file):
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if tmp_path.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif tmp_path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(tmp_path)
        else:
            raise ValueError("❌ Unsupported file type")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)
    finally:
        os.remove(tmp_path)

def embed_and_store(docs):
    Pinecone.from_documents(
        documents=docs,
        embedding=embedding_model,
        index_name=index_name,
        text_key="text"
    )

def ingest_document(uploaded_file):
    with NamedTemporaryFile(delete=True, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
        file_path = tmp.name

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        else:
            return "❌ Unsupported file type."

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        Pinecone.from_documents(
            documents=chunks,
            embedding=embedding_model,
            index_name=index_name,
            text_key="text"
        )

        return f"✅ Ingested {len(chunks)} chunks into Pinecone."

def generate_query_variants(query: str):
    prompt = f"""
    Rephrase the following question into 3 diverse queries for document retrieval.
    Format STRICTLY as a Python list: ["...", "...", "..."]

    Question: {query}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.strip()}]
    )
    raw_output = response.choices[0].message.content.strip()
    try:
        queries = ast.literal_eval(raw_output)
        assert isinstance(queries, list) and all(isinstance(q, str) for q in queries)
    except Exception as e:
        print("⚠️ Could not parse rephrased output:", e)
        queries = [query]

    return {"queries": queries, "original": query, "query": query}

def retrieve_documents(state):
    print("🧪 RAW query object:", state["query"], type(state["query"]))

    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embedding_model, text_key="text")

    raw_query = state["query"]

    if isinstance(raw_query, dict):
        query = raw_query.get("query", "")
    else:
        query = raw_query

    if not isinstance(query, str):
        raise ValueError(f"Expected query to be string, got {type(query)}: {query}")

    print("🔍 Final query string used for similarity search:", query)
    docs = vectorstore.similarity_search(query, k=3)

    return {
        "retrieved_docs": docs,
        "original": state["original"]
    }

def fuse_and_summarize(state):
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    query = state["original"]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "retrieved_docs": state["retrieved_docs"]
    }