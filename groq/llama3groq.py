import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

groq_api = os.getenv("GROQ_API")

st.title("Llama3 with Groq API")

llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
        Answer the question based on provided context only.
        Please provide the most accurate response based on question
        <context>{context}</context>
        Questions: {input} 
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

prompt1 = st.text_input("Enter your questions")

if st.button("Load Documets"):
    vector_embedding()
    st.write("Vector DB is ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['answer']):
            st.write(doc.page_content)
            st.write("--------------------------------------")
