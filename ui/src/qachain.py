from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.combine_documents.stuff import StuffDocumentsChain, LLMChain

# from src.CustomGPTCache import CustomGPTCache
import os
from dotenv import load_dotenv
from langchain.vectorstores.deeplake import DeepLake
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, 
    ChatGoogleGenerativeAI
)
gemini_api_key = st.secrets["gemini_api_key"]
cache_threshold = st.secrets["CACHE_THRESHOLD"]
llm_temperature = st.secrets["LLM_TEMPERATURE"]

# load_dotenv()
class QAChain:
    def __init__(self, model_usage,vector) -> None:
        # Initialize Gemini Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key,
            task_type="retrieval_query",
        )

        # Initialize Gemini Chat model
        self.model = ChatGoogleGenerativeAI(
            # model="models/gemini-1.5-pro-latest",
            model= str(model_usage),
            temperature= float(llm_temperature),
            google_api_key=gemini_api_key,
            convert_system_message_to_human=True,
        )
        self.vector = vector
        # Initialize GPT Cache
        # self.cache = CustomGPTCache()
        self.text_vectorstore = None
        self.text_retriever = None

    # def ask_question(self, query):
    #     try:
    #         # Search for similar query response in cache
    #         cached_response = self.cache.find_similar_query_response(
    #             query=query, threshold=int(cache_threshold)
    #         )

    #         # If similar query response is present,vreturn it
    #         if len(cached_response) > 0:
    #             print("Using cache")
    #             result = cached_response[0]["response"]
    #         # Else generate response for the query
    #         else:
    #             print("Generating response")
    #             result = self.generate_response(query=query)
    #     except Exception as _:
    #         print("Exception raised. Generating response.")
    #         result = self.generate_response(query=query)

    #     return result

    def generate_response(self, query: str):
        # Initialize the vectorstore and retriever object
        vstore = self.vector
        retriever = vstore.as_retriever(search_type="similarity")
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 15
        retriever.search_kwargs["k"] = 10

        # Write prompt to guide the LLM to generate response
        prompt_template = """You are helpful assistant of Bank Sinarmas, answer in 'Bahasa indonesia',
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}\n
        Question:\n {question}\n

        Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        document_prompt = PromptTemplate(
            input_variables=["page_content","file_name", "page_no"],
            template="Context:\npage content{page_content},\nfile name :{file_name} \npage number:{page_no}",
        )
        
        lc = LLMChain(llm=self.model, prompt=PROMPT)
        handler = StdOutCallbackHandler()
        combine_documents = StuffDocumentsChain(
        llm_chain=lc,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=[handler],
        )

        # chain_type_kwargs = {"prompt": PROMPT}
        

        # Create Retrieval QA chain
        qa = RetrievalQA(
            combine_documents_chain=combine_documents,
            retriever=retriever,
            # verbose=False,
            # chain_type_kwargs=chain_type_kwargs,
            return_source_documents =  True,
            callbacks=[handler],
        )

        # Run the QA chain and store the response in cache
        result = qa(query)
        answer = result['result']
        searchDocs = vstore.similarity_search(result['result'])
        metadata = [j.metadata for j in searchDocs][0]
        answer_with_source =  f"""{answer}\n\n
Sumber File : {metadata['file_name']} \n\nHalaman : {metadata['page_no']} """
        # self.cache.cache_query_response(query=query, response=result)
        
        return answer_with_source