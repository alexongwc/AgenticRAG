# src/app.py
import streamlit as st
from ingestion_graph import ingestion_graph
from query_graph import query_graph
from agents import generate_query_variants, load_and_split_docs
import openai

# Set API key for OpenAI
openai.api_key = st.secrets.get("OPENAI_API_KEY")
print("🔑 DEBUG: OPENAI_API_KEY =", "FOUND" if openai.api_key else "MISSING")

st.set_page_config(page_title="Agentic RAG ", layout="centered")
st.title("Agentic RAG ")
st.write("Upload a PDF or Excel file and ask a question about its content.")

uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    with st.spinner("Processing and embedding file..."):
        docs = load_and_split_docs(uploaded_file)
        ingestion_graph.invoke({"docs": docs})
        st.success("File embedded into Pinecone (file not stored).")

    st.subheader("Ask questions about the uploaded file")

    for pair in st.session_state.chat_history:
        st.markdown(f"**You:** {pair.get('query', '')}")
        st.markdown(f"**Bot:** {pair.get('answer', 'No answer generated.')}")

        if "sources" in pair and pair["sources"]:
            with st.expander("Sources used"):
                for j, doc in enumerate(pair["sources"]):
                    st.markdown(f"**Source {j + 1}:**")
                    st.markdown(f"> {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
                    if doc.metadata:
                        st.caption(f"Metadata: {doc.metadata}")

    user_query = st.text_input("Your question:", key="user_query_input")

    if user_query and st.session_state.get("last_question") != user_query:
        with st.spinner("Generating answer..."):
            query_state = generate_query_variants(user_query)
            result = query_graph.invoke(query_state)

            st.session_state.chat_history.append({
                "query": user_query,
                "answer": result.get("answer", "No answer returned"),
                "sources": result.get("retrieved_docs", [])
            })

            st.session_state.last_question = user_query
        st.rerun()