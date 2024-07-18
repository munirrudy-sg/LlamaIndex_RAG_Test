from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.CustomGPTCache import CustomGPTCache
import os
from dotenv import load_dotenv
from langchain.vectorstores.deeplake import DeepLake
import streamlit as st
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, 
    ChatGoogleGenerativeAI
)

google_api_key = st.secrets["gemini_api_key"]
cache_threshold = st.secrets["CACHE_THRESHOLD"]

load_dotenv()
class QAChain:
    def __init__(self, model_usage) -> None:
        # Initialize Gemini Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key,
            task_type="retrieval_query",
        )

        # Initialize Gemini Chat model
        self.model = ChatGoogleGenerativeAI(
            # model="models/gemini-1.5-pro-latest",
            model= str(model_usage),
            temperature=0.3,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
        )

        # Initialize GPT Cache
        self.cache = CustomGPTCache()
        self.text_vectorstore = None
        self.text_retriever = None

    def ask_question(self, query):
        try:
            # Search for similar query response in cache
            cached_response = self.cache.find_similar_query_response(
                query=query, threshold=int(cache_threshold)
            )

            # If similar query response is present,vreturn it
            if len(cached_response) > 0:
                print("Using cache")
                result = cached_response[0]["response"]
            # Else generate response for the query
            else:
                print("Generating response")
                result = self.generate_response(query=query)
        except Exception as _:
            print("Exception raised. Generating response.")
            result = self.generate_response(query=query)

        return result

    def generate_response(self, query: str):
        # Initialize the vectorstore and retriever object
        vstore = DeepLake(
            dataset_path="../../database/text_vectorstore",
            embedding=self.embeddings,
            read_only=True,
            num_workers=4,
            verbose=False,
        )
        retriever = vstore.as_retriever(search_type="similarity")
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 20
        retriever.search_kwargs["k"] = 15

        # Write prompt to guide the LLM to generate response
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}\n
        Question:\n {question}\n

        Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        print(PROMPT)

        chain_type_kwargs = {"prompt": PROMPT}

        # Create Retrieval QA chain
        qa = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=retriever,
            verbose=True,
            chain_type_kwargs=chain_type_kwargs,
        )

        # Run the QA chain and store the response in cache
        result = qa({"query": query})["result"]
        # QA_query =  qa({"query": query})
        self.cache.cache_query_response(query=query, response=result)
        
        return result