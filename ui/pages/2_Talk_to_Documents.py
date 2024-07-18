import streamlit as st
import os
import tempfile
from pathlib import Path
import shutil
from ingestion import Ingestion
from src.qachain import QAChain

# extracting text from document
def get_document_text(doc):
    ingestion = Ingestion()
    ingestion.ingest_documents(doc)

def get_conversationchain(query,selected_option):
    if selected_option == "gemini-1.5-pro-latest":
        model = "models/gemini-1.5-pro-latest"
    else:
        model = "models/gemini-1.5-flash"

    qna = QAChain(model)
    results = qna.ask_question(
        query=query
    )
    return results

def clear_vector_db():
    st.session_state.messages = [{"role": "assistant", "content": "upload some documents and ask me a question"}]
    abs_path = os.path.dirname(os.path.abspath(__file__))
    CurrentFolder = str(Path(abs_path).resolve())
    path = os.path.join(CurrentFolder, "database")
    shutil.rmtree(path)

# generating response from user queries and displaying them accordingly
# def handle_question(question):
#     response=st.session_state.conversation({'question': question})
#     st.session_state.chat_history=response["chat_history"]
#     for i,msg in enumerate(st.session_state.chat_history):
#         if i%2==0:
#             st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some documents and ask me a question"}]
    
def main():
    st.set_page_config(page_title="Chat with multiple DOCUMENTs",page_icon="🤖")


    with st.sidebar:
        st.title("Menu:")
        st.subheader("Your documents")
        # docs=st.file_uploader("Upload your document here and click on 'Process'",accept_multiple_files=True)
        # Create the dropdown
        options = ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
        selected_option = st.selectbox("Select Gemini Model:", options, index= 1)

        docs = st.file_uploader("File upload", type= ['pdf', 'docx'] ,accept_multiple_files=True)
        # print(selected_option)
        if st.button("Process"):
            with st.spinner("Processing"):
                if docs:
                    for doc in docs:
                        temp_dir = tempfile.mkdtemp()
                        path = os.path.join(temp_dir, doc.name)
                        print(path)
                        with open(path, "wb") as f:
                            f.write(doc.getvalue())
                        print("success")
                        #extract from document -> get the text chunk -> create vectore store
                        get_document_text(path)
                    st.success("Done")
        
    # Main content area for displaying chat messages
    st.title("Chat with document files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    st.sidebar.button('Clear VectorDB', on_click=clear_vector_db)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some documents and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_conversationchain(prompt,selected_option)
                placeholder = st.empty()
                full_response = ''
                for item in list(response):
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == '__main__':
    main()