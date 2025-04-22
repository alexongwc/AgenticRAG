# src/query_graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document
from agents import generate_query_variants, retrieve_documents, fuse_and_summarize

# Define the state schema for LangGraph
class QueryState(TypedDict):
    query: str
    queries: List[str]
    retrieved_docs: List[Document]
    original: str
    answer: str

# Pass the schema to StateGraph
graph_builder = StateGraph(QueryState)

# Add nodes (LangGraph agents)
graph_builder.add_node("generate_queries", generate_query_variants)
graph_builder.add_node("retrieve_docs", retrieve_documents)
graph_builder.add_node("generate_response", fuse_and_summarize)

# Define flow
graph_builder.set_entry_point("generate_queries")
graph_builder.add_edge("generate_queries", "retrieve_docs")
graph_builder.add_edge("retrieve_docs", "generate_response")
graph_builder.set_finish_point("generate_response")

# Compile the graph
query_graph = graph_builder.compile()
