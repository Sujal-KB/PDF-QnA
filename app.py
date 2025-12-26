import streamlit as st
from rag import *

st.set_page_config(page_title="PDF QnA", layout="centered")
st.title("📄 RAG Application for PDF QnA")
st.info("Upload a PDF and ask questions related to its content")

@st.cache_resource
def process_pdf(pdf_file):
    text = load_pdf(pdf_file)
    chunks = create_chunk(text)
    index, chunk_mapping = create_index(chunks)
    return index, chunk_mapping

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    index, chunk_mapping = process_pdf(pdf)

    query = st.text_input("Ask a question from the PDF")

    if st.button("Get Answer"):
        with st.spinner("Retriving Information..."):
            top_chunks = retrive_top_k(chunk_mapping, index, query)
            prompt = build_prompt(top_chunks, query)
            answer = generate_completion(prompt)

            st.subheader("Answer")
            st.write(answer)
