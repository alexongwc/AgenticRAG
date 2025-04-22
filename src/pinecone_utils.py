# 📁 src/pinecone_utils.py
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings

# Load from Streamlit secret
api_key = st.secrets.get("PINECONE_API_KEY")
env = st.secrets.get("PINECONE_ENV")
index_name = st.secrets.get("PINECONE_INDEX")
openai_key = st.secrets.get("OPENAI_API_KEY")

# Init Pinecone
pc = Pinecone(api_key=api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=env)
    )

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

__all__ = ["embedding_model", "index_name"]
