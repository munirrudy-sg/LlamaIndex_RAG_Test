import os
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
import streamlit as st
from langchain.text_splitter import (
    CharacterTextSplitter,
)

document_charsplitter_chunksize = st.secrets["DOCUMENT_CHARSPLITTER_CHUNKSIZE"]
document_charsplitter_chunk_overlap = st.secrets["DOCUMENT_CHARSPLITTER_CHUNK_OVERLAP"]

load_dotenv()
class DOCUMENTReader:

    def __init__(self) -> None:
        self.file_name = ""
        self.total_pages = 0

    def load_document(self, file_path):
        # Get the filename from file path
        self.file_name = os.path.basename(file_path)
        
        if self.file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = PyPDFLoader(file_path)

        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=int(document_charsplitter_chunksize),
            chunk_overlap=int(document_charsplitter_chunk_overlap),
        )

        # Load the pages from the document
        pages = loader.load()
        self.total_pages = len(pages)
        chunks = []
        
        # Loop through the pages
        for idx, page in enumerate(pages):
            # Append each page as Document object with modified metadata
            chunks.append(
                Document(
                    page_content=page.page_content,
                    metadata=dict(
                        {
                            "file_name": self.file_name,
                            "page_no": str(idx + 1),
                            "total_pages": str(self.total_pages),
                        }
                    ),
                )
            )

        # Split the documents using splitter
        final_chunks = text_splitter.split_documents(chunks)
        return final_chunks