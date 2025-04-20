# src/pinecone_utils.py

import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

# Ensure old env is cleared
os.environ.pop("OPENAI_API_KEY", None)

# Load .env from root
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path)

# 🔍 Confirm loaded key
print("✅ Loaded API key:", repr(os.getenv("OPENAI_API_KEY")))

# Init Pinecone v3
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=env)
    )

# Do NOT instantiate the index here
# Just pass the index_name to LangChain
# LangChain will handle `pc.Index(...)` internally

# ✅ Instantiate embeddings AFTER dotenv
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
