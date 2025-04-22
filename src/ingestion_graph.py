from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document
from agents import embed_and_store

# State only includes pre-processed documents
class IngestionState(TypedDict):
    docs: List[Document]

builder = StateGraph(IngestionState)

# Only one node: embed into Pinecone
def embed_and_store_node(state: IngestionState):
    embed_and_store(state["docs"])
    return {}

builder.add_node("embed_and_store", embed_and_store_node)
builder.set_entry_point("embed_and_store")
builder.set_finish_point("embed_and_store")

ingestion_graph = builder.compile()
