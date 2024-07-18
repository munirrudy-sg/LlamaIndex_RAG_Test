import os
from src.document_reader import DOCUMENTReader
from langchain.vectorstores.deeplake import DeepLake
import streamlit as st
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)


google_api_key = st.secrets["gemini_api_key"]
cache_threshold = st.secrets["CACHE_THRESHOLD"]

class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key,
        )

    def ingest_documents(
        self,
        file: str,
    ):
        
        loader = DOCUMENTReader()
        chunks = loader.load_document(file_path=file)
        
        # Initialize the vector store
        vstore = DeepLake(
            dataset_path="database/text_vectorstore",
            embedding=self.embeddings,
            overwrite=True,
            num_workers=4,
            verbose=False,
        )
        
        # Ingest the chunks
        _ = vstore.add_documents(chunks)