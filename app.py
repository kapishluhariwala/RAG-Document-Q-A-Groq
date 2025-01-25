import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'retriever' not in st.session_state:
    st.session_state.retriever = []

st.set_page_config(page_title="RAG Q&A Chatbot", layout = 'wide')
st.title('RAG Q&A Chatbot with GROQ')

st.sidebar.title('Settings')
groq_api_key = st.sidebar.text_input('GROQ API Key', type='password')
hf_token = st.sidebar.text_input('HuggingFace Toekn', type='password')

llm_select = st.sidebar.selectbox('GROQ Language Model', ['mixtral-8x7b-32768', 'llama3-70b-8192', 'llama3-8b-8192'])

temperature = st.sidebar.slider('Temperature', 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider('Max Tokens', 50, 300, 150)

llm = ChatGroq(model=llm_select,api_key=groq_api_key,temperature=temperature,max_tokens=max_tokens)

prompt = ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context if available else 
    you can answer the user query.Please provide the most accuracte response 
    sbased on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    '''
)

uploaded_files = st.sidebar.file_uploader('Please upload your files', accept_multiple_files=True, type='pdf')
if(uploaded_files):
    if(st.sidebar.button('Create Embeddings')):
        st.toast(f'Number of files uploaded: {len(uploaded_files)}')
        st.toast('Please wait while files are being uploaded')

        if 'processed_data' not in st.session_state:
            documents = []
            with st.spinner("Processing... Please wait."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(os.getcwd(),'temp', uploaded_file.name)

                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()

                documents.extend(loaded_docs)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                document_chunks = text_splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
                vectors = FAISS.from_documents(documents, embeddings)

                st.session_state.retriever = vectors.as_retriever()
                st.toast('Embeddings created')


output_parser = StrOutputParser()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
    
if user_prompt := st.chat_input('Say Something...'):
    st.session_state.messages.append({'role':'user', 'content':user_prompt})
    with st.chat_message('user'):
        st.markdown(user_prompt)
    
    if uploaded_files:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(st.session_state.retriever,document_chain)
        response = retrieval_chain.invoke({'input': user_prompt})['answer']


    else:
        chain = prompt|llm|output_parser
        response = chain.invoke({'context':'','input':user_prompt})

    
    with st.chat_message("assistant"):
        st.markdown(response)    
    st.session_state.messages.append({"role": "assistant", "content": response})




