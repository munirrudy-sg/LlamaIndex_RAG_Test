import os
import json
import streamlit as st
import sys
import path
from langchain_community.vectorstores import FAISS
from src.examples import query
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector


db_gemini_api_key = st.secrets["db_gemini_api_key"]

# dir = path.Path(__file__).abspath()
# sys.path.append(dir.parent.parent)



class Prep:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.query = query
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=db_gemini_api_key,
        )

    def get_query_examples(
        self
    ):

        # # Open and read the JSON file
        # with open(path_to_query, 'r') as json_file:
        #     examples = json.load(json_file)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.query,
            self.embeddings,
            FAISS,
            k=5,
            input_keys=["input"],
        )

        return example_selector