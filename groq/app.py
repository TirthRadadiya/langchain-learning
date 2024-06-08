import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv

load_dotenv()

groq_api = os.environ["GROQ_API"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
    st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("Chat using Groq API")
llm = ChatGroq(groq_api=groq_api, model_name="Mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>{context}</context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriver = st.session_state.vectors.as_retriever()

retrieval_chain = create_retrieval_chain(retriver, document_chain)

prompt = st.text_input("Ask your question here...")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start_time}")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['answer']):
            st.write(doc.page_content)
            st.write("--------------------------------------")