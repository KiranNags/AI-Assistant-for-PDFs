import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import tempfile

# Try to initialize OpenAI embeddings, fall back to HuggingFace if quota is exceeded
def get_embeddings(openai_key=None):
    try:
        # If OpenAI key is provided, use OpenAI Embeddings
        if openai_key:
            return OpenAIEmbeddings(openai_api_key=openai_key)
        else:
            raise ValueError("API key not provided")
    except Exception as e:
        print(f"Error with OpenAI: {e}")
        print("Switching to HuggingFace embeddings.")
        # Use HuggingFace embeddings if OpenAI fails
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="PDF Q&A Tool", layout="centered")
st.title("Ask your PDF anything")

openai_key = st.text_input("Enter your OpenAI API Key (leave blank for HuggingFace)", type="password")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully! Initializing...")

    # Save uploaded file locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Create temporary DB directory
    persist_directory = tempfile.mkdtemp()

    # Get the embeddings (either OpenAI or HuggingFace)
    embeddings = get_embeddings(openai_key)

    # Create Chroma DB with embeddings
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_key),
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa.run(question)
        st.markdown(f"**Answer:** {answer}")
