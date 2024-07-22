import os
from src.document_reader import DOCUMENTReader
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)

gemini_api_key = st.secrets["gemini_api_key"]


class Prep:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key,
        )

    def ingest_documents(
        self,
        file: str,
    ):
        
        loader = DOCUMENTReader()
        chunks = loader.load_document(file_path=file)
        
        # Initialize the vector store
        vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        return vector_store