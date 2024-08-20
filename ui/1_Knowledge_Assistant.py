__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import sqlite3
import streamlit as st
from streamlit_feedback import streamlit_feedback
import uuid
from datetime import datetime

import chromadb
import speech_recognition as sr
from io import BytesIO

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

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

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
# collection_name = "promosi-bsim-20240718"

# # Load from disk
# load_client = chromadb.PersistentClient(path="./chroma_db")

# # Fetch the collection
# chroma_collection = load_client.get_collection(collection_name)

# configuration = {
#     "client_type": "PersistentClient",
#     "path": "./chroma_db"
# }
# # Fetch the vector store
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# # Get the index from the vector store
# index = VectorStoreIndex.from_vector_store(
#     vector_store,
#     similarity_top_k=7
# )

## For Milvus
vector_store = MilvusVectorStore(
    uri="./milvus_db/milvus_20240820.db",
    dim=768,
    overwrite=False,
    collection_name = "bsim_web_20240820"
    # enable_sparse=True, # uncomment for hybrid
    # hybrid_ranker="RRFRanker",
    # hybrid_ranker_params={"k": 100},
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

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
    * Only use the provided context to answer; do not be generative!


    **Additional Considerations**
    *'list outlet' is the restaurant/place/city in indonesia where the promo is eligible,
    *There are two types of credit card in Bank Sinarmas 'personal' and 'korporat'. 
    *'personal' credit card have 'silver' and 'platinum' category. Both of them have free annual fee for lifetime.
    *'korporat' credit card have only 'platinum' category.
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

    # query_engine = index.as_query_engine(text_qa_template=llm_prompt, similarity_top_k=7, llm=llm, response_mode="simple_summarize", vector_store_query_mode="hybrid") # for hybrid
    query_engine = index.as_query_engine(text_qa_template=llm_prompt, similarity_top_k=7, llm=llm, response_mode="simple_summarize")
    response = query_engine.query(prompt_input)
    print(response)

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
def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening... Please speak into the microphone.")
        audio = recognizer.listen(source)
    
    try:
        st.info("Recognizing speech...")
        text = recognizer.recognize_google(audio, language='id-ID')
        return text
    except sr.RequestError:
        return "API was unreachable or unresponsive."
    except sr.UnknownValueError:
        return "Unable to recognize speech."

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  question TEXT, 
                  response TEXT, 
                  response_type TEXT, 
                  user_feedback TEXT,
                  timestamp TEXT, 
                  session_id TEXT)''')  # Store session_id
    conn.commit()
    conn.close()

# Function to store feedback in the SQLite database
def store_feedback(question, response, response_type, user_feedback, timestamp, session_id):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''INSERT INTO feedback (question, response, response_type, user_feedback, timestamp, session_id) VALUES (?, ?, ?, ?, ?, ?)''', 
              (question, response, response_type, user_feedback, timestamp, session_id))
    conn.commit()
    conn.close()

# Feedback mechanism using streamlit-feedback
def handle_feedback(user_response, result):
    st.write(f"Session ID: {st.session_state.session_id}")
    # st.write("Result:", response)
    st.write(f"User feedback: {user_response}")
    st.toast("✔️ Feedback received!")

    response_type = 'good' if user_response['score']=='👍' else 'bad'
    feedback = user_response['text']
    # Get the current timestamp
    timestamp = datetime.now()

    # Format the timestamp as a string in 'yyyymmddhhmmss' format
    timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')

    # Store feedback in the database
    store_feedback(st.session_state.messages[-2]["content"], result, response_type, feedback, timestamp_str, st.session_state.session_id)
        

    # Reset session ID after feedback is submitted
    st.session_state["session_id"] = str(uuid.uuid4())

# Main function for Streamlit app
def main():
    # Initialize the feedback database
    init_db()

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())  # Generate unique session ID for each user

    # App title
    st.set_page_config(page_title="AIDSU Chatbot RAG👩‍🦰💬")
    st.title('AIDSU - Bank Sinarmas Knowledge')

    with st.sidebar:
        options = ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
        selected_option = st.selectbox("Select Gemini Model:", options, index=0)
        st.write("""Elisa bisa bantu kamu mengetahui informasi promo dan layanan berikut:  
        \n1. Info Promosi  
        \n2. Info Produk Tabungan  
        \n3. Info Produk Deposito  
        \n4. dll  
        \n\nContoh pertanyaan:  
        \n1. Promo apa saja yang tersedia?  
        \n2. Cara daftar cc korporat?  
        \n3. List promo yang tersedia di medan!  
        \n4. Apakah promo magal bisa pakai cc korporat?""")
        st.button('Clear Chat History', on_click=clear_chat_history)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Option for Voice or Text Input
    input_method = st.radio("Choose your input method:", ("Text", "Voice"))

    if input_method == "Voice":
        if st.button("Start Listening"):
            prompt = recognize_speech_from_microphone()
            st.write("You said: ", prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

    elif input_method == "Text":
        # Text input option
        if prompt := st.chat_input("Type your question here:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm_response = generate_gemini_response(preprocess_input(st.session_state.messages[-1]["content"]), selected_option)
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

        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
        
    full_response = st.session_state.messages[-1]['content']
    # print(st.session_state.messages[-1])
    # Feedback submission form
    feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                key=st.session_state.session_id,
                on_submit=handle_feedback,
                kwargs={"result": full_response},
            )
    print(feedback)
if __name__ == "__main__":
    main()
