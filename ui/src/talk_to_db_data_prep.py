import os
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector


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

    def get_query_examples(
        self
    ):
        
        # Define the path to the JSON file
        file_path = 'src/examples.json'

        # Open and read the JSON file
        with open(file_path, 'r') as json_file:
            examples = json.load(json_file)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            self.embeddings,
            FAISS,
            k=10,
            input_keys=["input"],
        )

        return example_selector