# Importing libraries
import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

# API and model setting
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit app
st.title("𝖯𝖣𝖥𝖲𝖾𝗇𝗌𝖾 : 𝖯𝖣𝖥 𝖰𝗎𝖾𝗌𝗍𝗂𝗈𝗇 𝖺𝗇𝖽 𝖠𝗇𝗌𝗐𝖾𝗋𝗂𝗇𝗀 𝗐𝗂𝗍𝗁 𝗌𝖾𝗌𝗌𝗂𝗈𝗇 𝖼𝗁𝖺𝗍 𝗁𝗂𝗌𝗍𝗈𝗋𝗒")
st.write("upload pdfs and ask questions related to pdfs")
llm=ChatGroq(model="Gemma2-9b-It")
session_id=st.text_input("Session id",value="common_session")

# manage chat history
if 'store' not in st.session_state:
    st.session_state.store={}

# Upload files and documents loading
uploaded_files=st.file_uploader("Drop the pdf files here",type="pdf",accept_multiple_files=True)
if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
        docs=PyPDFLoader(temppdf).load()
        documents.extend(docs)
    # Delete the temp file as we no longer need it
    if os.path.exists("./temp.pdf"):
        os.remove("./temp.pdf")
    # Text splitting and embedding and storing in chromadb
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits=text_splitter.split_documents(documents)
    faiss_index = FAISS.from_documents(splits, embeddings)
    retriever=faiss_index.as_retriever()

    # Prompts
    context_system_prompt=(
        "Given a chat history and latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do Not answer the question, "
        "just reformulate it if needed and otherwise return it as it is"
    )
    context_prompt=ChatPromptTemplate.from_messages([
        ("system",context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")]
    )

    history_aware_ret=create_history_aware_retriever(llm,retriever,context_prompt)

    system_prompt=(
        "You are 'PDFSense' a PDF reading and answering assistant. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you dont know."
        "Answer the questions nicely."
        "\n\n"
        "{context}"
    )

    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    # Chain for the chatbot
    qa_chain=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(history_aware_ret,qa_chain)

    # Session Id storing in chat history
    def get_session_history(session:str)-> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    # RAG with history
    conversation_rag=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"                                        
        )
    
    user_input=st.text_input("Enter question")
    if user_input:
        session_history=get_session_history(session_id)
        response=conversation_rag.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}
            },
        )
        st.write(st.session_state.store)
        st.write("Assistant:",response['answer'])
        st.write("Chat History",session_history.messages)