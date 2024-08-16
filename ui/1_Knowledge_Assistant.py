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
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import yaml

from src.utils import preprocess_input

temperature = st.secrets['knowledge_assistant_tmp']
generation_config = {"temperature": temperature}
# safety_settings = 

gemini_api_key = st.secrets["gemini_api_key"]
gemini_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/embedding-001")

# Set Global settings
# Settings.llm = model
Settings.embed_model = gemini_embedding_model
Settings.context_window = 50000

# Load the YAML data
config = yaml.safe_load(open('config/config.yaml', 'r'))

### For Chroma
collection_name = "promosi-bsim-20240718"

# Load from disk
load_client = chromadb.PersistentClient(path="./chroma_db")

# Fetch the collection
chroma_collection = load_client.get_collection(collection_name)

configuration = {
    "client_type": "PersistentClient",
    "path": "./chroma_db"
}
# Fetch the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Get the index from the vector store
index = VectorStoreIndex.from_vector_store(
    vector_store,
    similarity_top_k=7
)

## For Milvus
# vector_store = MilvusVectorStore(
#     uri="./milvus_db/milvus_vdb_bsim_20240801.db",
#     dim=768,
#     overwrite=False,
#     enable_sparse=True,
#     hybrid_ranker="RRFRanker",
#     hybrid_ranker_params={"k": 100},
# )
# index = VectorStoreIndex.from_vector_store(vector_store)

template = ("""You are a knowledgeable and friendly virtual assistant of Bank Sinarmas, aiming to provide exceptional customer service.
    Leverage the provided context to tailor your responses accurately and provide exact information with context.
    Strive to understand the user's underlying needs and goals to provide the most helpful response.
    Use clear, concise, and polite indonesian language in your responses. Avoid jargon, technical terms, assumptions, and generalizations.
    If you encounter an ambiguous query or lack sufficient information, politely ask for clarification.
    The context were about bank sinarmas profile, management/stakeholder, product and promotion.

    **Key points to remember:**
    * Prioritize customer satisfaction.
    * Offer additional assistance or information when possible.
    * Use a conversational and engaging tone.
    * Maintain a professional demeanor.
    * Only use provided context to answer, do not generative!


    **Additional Considerations**
    *'list outlet' is the restaurant/place/city in indonesia where the promo is eligible,
    *There are two types of credit card in Bank Sinarmas 'personal' and 'korporat'.
    *If the promo only contain one type of credit card, then the other is not eligible.
    *Bancassurance is similar with 'asuransi'.
    *Data or context is provided from Bank Sinarmas website.
    *Bank Sinarmas website: https://www.banksinarmas.com/id/.
    *Bank Sinarmas call center: 1500153.

Question: {query_str} \nContext: {context_str} \nAnswer:"""
)

llm_prompt = PromptTemplate(template)
opening_content = """Selamat pagi/siang/sore/malam! 👋  \nApa yang ingin Anda tanyakan tentang Bank Sinarmas?  \nKami siap membantu Anda dengan informasi mengenai:  \n1. Profil Bank Sinarmas.  \n2. Manajemen Bank Sinarmas.  \n3. Promo yang tersedia.  \n4. Produk-produk Bank Sinarmas."""

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": f"{opening_content}"}]

def generate_gemini_response(prompt_input, selected_option):

    if selected_option == "gemini-1.5-pro-latest":
        model_name = "models/gemini-1.5-pro-latest"
    else:
        model_name = "models/gemini-1.5-flash"

    llm = Gemini(api_key=gemini_api_key, model_name=model_name, generation_config=generation_config)

    query_engine = index.as_query_engine(text_qa_template=llm_prompt, similarity_top_k=10, llm=llm, response_mode="simple_summarize", vector_store_query_mode="hybrid")

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
    # unique_titles = unique_titles[:5]
    text = ''
    for title in unique_titles:
        url = get_url_from_title(config, title)
        text += url + "  \n"

    return text


def main():
    # App title
    st.set_page_config(page_title="AIDSU Chatbot RAG👩‍🦰💬")
    st.title('AIDSU - Bank Sinarmas Knowledge')

    # Clear chat everytime pages move
    # clear_chat_history()
    
    with st.sidebar:
        options = ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
        selected_option = st.selectbox("Select Gemini Model:", options, index= 0)
        st.write("""Elisa bisa bantu kamu mengetahui informasi promo dan layanan berikut:  \n1. Info Promosi  \n2. Info Produk Tabungan  \n3. Info Produk Deposito  \n4. dll  \n\nContoh pertanyaan:  \n1. Promo apa saja yang tersedia?  \n2. Cara daftar cc korporat?  \n3. List promo yang tersedia di medan!  \n4. Apakah promo magal bisa pakai cc korporat?""" 
)
        # Main content area for displaying chat messages
        st.button('Clear Chat History', on_click=clear_chat_history)


    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": f"{opening_content}"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm_response = generate_gemini_response(preprocess_input(prompt), selected_option)
                response = llm_response[0]
                metadata = llm_response[1]
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)

                sources = get_sources(metadata)
                full_response += "\n\n Sumber:  \n" + sources
                placeholder.markdown(full_response) 
            # with st.sidebar:
            #     st.write(f"\n Metadata: \n{metadata}")
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

if __name__ == '__main__':
    main()