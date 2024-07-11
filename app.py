import streamlit as st
from bs4 import BeautifulSoup

from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate

from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection


from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

gemini_api_key = st.secrets["gemini_api_key"]
gemini_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/embedding-001")
model = Gemini(api_key=gemini_api_key, model_name="models/gemini-1.5-flash")

# Set Global settings
Settings.llm = model
Settings.embed_model = gemini_embedding_model
Settings.chunk_size = 3000
Settings.chunk_overlap = 64
Settings.context_window = 30000


# # Load from disk
# load_client = chromadb.PersistentClient(path="chroma_db")

# # Fetch the collection
# chroma_collection = load_client.get_collection("promosi-bsim-20240710-v2")

configuration = {
    "client_type": "PersistentClient",
    "path": "/tmp/.chroma"
}

collection_name = "promosi-bsim-20240710-v2"

conn = st.experimental_connection("chromadb",
                                type=ChromaDBConnection,
                                **configuration)
documents_collection_df = conn.get_collection_data(collection_name)

# Fetch the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Get the index from the vector store
index = VectorStoreIndex.from_vector_store(
    vector_store,
    similarity_top_k=7
)

template = (
    """You are a helpful assistant of Bank Sinarmas, answer in 'Bahasa Indonesia'.
    You do not respond as 'User' or pretend to be 'User'. 
    The question is from user that want to know mostly about promotion in Bank Sinarmas
    Check all of provided context, the context is in bahasa indonesia
    Notes: 'list outlet' is the restaurant/place/city in indonesia where promo is eligible, 
            There are two types of credit card in Bank Sinarmas 'personal' and 'korporat'.
            If the promo only contain one type of credit card, then the other is not eligible.
            Data or context is provided from Bank Sinarmas website.
            website bank sinarmas 'https://www.banksinarmas.com/id/'
Question: {query_str} \nContext: {context_str} \nAnswer:"""
)

llm_prompt = PromptTemplate(template)

# App title
st.set_page_config(page_title="Prissa Promotion Chatbot RAG👩‍🦰💬")

# Replicate Credentials
with st.sidebar:
    st.title('Prissa Promotion Chatbot RAG👩‍🦰💬')
    st.write('This chatbot is created using the Gemini API LLM model from Google.')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_gemini_response(prompt_input):

    query_engine = index.as_query_engine(text_qa_template=llm_prompt, similarity_top_k=7, response_mode="simple_summarize")

    response = query_engine.query(prompt_input)

    # Step 7: Return the generated text
    return response.response

# Example usage:
# Assuming 'st.session_state.messages' is already populated with the required dialogue history.
# result = generate_gemini_response("What is the current promotion for the credit card?")
# print(result)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_gemini_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)