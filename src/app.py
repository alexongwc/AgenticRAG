import streamlit as st
import os
from dotenv import load_dotenv
from ingestion_graph import ingestion_graph
from query_graph import query_graph
from agents import generate_query_variants, load_and_split_docs
import openai

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 DEBUG: OPENAI_API_KEY =", repr(os.getenv("OPENAI_API_KEY")))

st.set_page_config(page_title="Agentic RAG MiniPoC", layout="centered")

# App title
st.title("📄 Agentic RAG Mini PoC")
st.write("Upload a PDF or Excel file and ask a question about its content.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File Ingestion ---
if uploaded_file is not None:
    with st.spinner("📥 Processing and embedding file..."):
        docs = load_and_split_docs(uploaded_file)
        ingestion_graph.invoke({"docs": docs})
        st.success("✅ File embedded into Pinecone (file not stored).")

    # --- Chat Interface ---
    st.subheader("💬 Ask questions about the uploaded file")

    # Render chat history
    for i, pair in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {pair.get('query', '🤔')}")
        st.markdown(f"**Bot:** {pair.get('answer', '⚠️ No answer generated.')}")

        if "sources" in pair and pair["sources"]:
            with st.expander("📚 Sources used"):
                for j, doc in enumerate(pair["sources"]):
                    st.markdown(f"**Source {j + 1}:**")
                    st.markdown(f"> {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
                    if doc.metadata:
                        st.caption(f"🔖 Metadata: {doc.metadata}")

    # Ask question
    user_query = st.text_input("Your question:", key="user_query_input")

    if user_query and st.session_state.get("last_question") != user_query:
        with st.spinner("🤖 Generating answer..."):
            query_state = generate_query_variants(user_query)
            print("🧪 Query state passed to LangGraph:", query_state)

            result = query_graph.invoke(query_state)
            print("✅ query_graph result =", result)

            st.session_state.chat_history.append({
                "query": user_query,
                "answer": result.get("answer", "⚠️ No answer returned."),
                "sources": result.get("retrieved_docs", [])
            })

            st.session_state.last_question = user_query

        # Optional: Clear input field after sending
        st.rerun()
