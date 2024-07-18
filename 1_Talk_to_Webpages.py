__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import streamlit as st

from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate

from llama_index.vector_stores.chroma import ChromaVectorStore


from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import yaml


from src.utils import preprocess_input

gemini_api_key = st.secrets["gemini_api_key"]
gemini_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/embedding-001")
model = Gemini(api_key=gemini_api_key, model_name="models/gemini-1.5-flash")

# Set Global settings
Settings.llm = model
Settings.embed_model = gemini_embedding_model
Settings.chunk_size = 3700
Settings.chunk_overlap = 0
Settings.context_window = 35000

collection_name = "promosi-bsim-20240718"

# Load the YAML data
config = yaml.safe_load(open('config/config.yaml', 'r'))

# Load from disk
load_client = chromadb.PersistentClient(path="./chroma_db")

# Fetch the collection
chroma_collection = load_client.get_collection(collection_name)

# configuration = {
#     "client_type": "PersistentClient",
#     "path": "./chroma_db"
# }



# conn = st.experimental_connection("chromadb",
#                                 type=ChromaDBConnection,
#                                 **configuration)

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
st.set_page_config(page_title="AIDSU Chatbot RAG👩‍🦰💬")

# Replicate Credentials
with st.sidebar:
    st.title('AIDSU Chatbot RAG👩‍🦰💬')
    st.write('This chatbot is created using the Gemini API LLM model from Google.')
    options = ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
    selected_option = st.selectbox("Select Gemini Model:", options, index= 1)

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
    return response.response, response.metadata

def get_url_from_title(yaml_data, search_title):
    for category, details in yaml_data.items():
        for detail_page in details['detail_pages']:
            if detail_page['title'] == search_title:
                return detail_page['url']
    return ''
        
def unique_preserve_order(input_list):
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def get_sources(metadata):
    # Extract the list of unique titles from metadata using a set
    titles = [info['title'] for info in metadata.values()]
    unique_titles = unique_preserve_order(titles)
    text = ''
    for title in unique_titles:
        url = get_url_from_title(config, title)
        text += url + "\n"

    return text

def get_conversationchain(selected_option):
    if selected_option == "gemini-1.5-pro-latest":
        model_name = "models/gemini-1.5-pro-latest"
    else:
        model_name = "models/gemini-1.5-flash"

    llm = Gemini(api_key=gemini_api_key, model_name=model_name)

    # Set llm
    Settings.llm = llm


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm_response = generate_gemini_response(preprocess_input(prompt))
            response = llm_response[0]
            metadata = llm_response[1]
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)

            sources = get_sources(metadata)
            full_response += "\n\n Sumber: \n\n" + sources
            placeholder.markdown(full_response) 
        with st.sidebar:
            st.write(f"\n Metadata: \n{metadata}")
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)