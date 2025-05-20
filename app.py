import streamlit as st
from scripts.rag_pipeline import load_rag_pipeline, query_rag
import re

st.title("Trading Q&A")
st.write("Ask questions about trading and get clear, accurate answers!")

# Load RAG pipeline
@st.cache_resource
def get_qa_chain():
    return load_rag_pipeline()

qa_chain = get_qa_chain()

# User input
question = st.text_input("Enter your question:", placeholder="e.g., What is fundamental analysis?")
if question:
    with st.spinner("Generating answer..."):
        answer, sources = query_rag(qa_chain, question)
        # Clean the answer to remove instructional text or unwanted artifacts
        clean_answer = re.sub(
            r"(Use the following pieces of context.*?\nAnswer:|\nAnswer:|\s*Answer:\s*)",
            "",
            answer,
            flags=re.DOTALL
        ).strip()
        # Remove any trailing instructional text or artifacts
        clean_answer = re.sub(r"\n\n.*", "", clean_answer, flags=re.DOTALL).strip()
        st.subheader("Answer")
        st.write(clean_answer)
        st.subheader("Sources")
        for doc in sources:
            st.write(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")
