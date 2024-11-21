# Importing libraries
import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API and model settings
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit app
st.title("PDFSense : PDF Question and Answering with Session Chat History")
st.write("Upload PDFs and ask questions related to the content of the PDFs.")
llm = ChatGroq(model="Gemma2-9b-It")
session_id = st.text_input("Session ID", value="common_session")

# Manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# Upload files and document loading
uploaded_files = st.file_uploader("Drop the PDF files here", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        docs = PyPDFLoader(temppdf).load()
        documents.extend(docs)

    # Delete the temp file as we no longer need it
    if os.path.exists("./temp.pdf"):
        os.remove("./temp.pdf")

    # Text splitting and embedding, storing in FAISS index
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    faiss_index = FAISS.from_documents(splits, embeddings)
    retriever = faiss_index.as_retriever()

    # Prompts
    context_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do not answer the question, "
        "just reformulate it if needed and otherwise return it as it is."
    )
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_ret = create_history_aware_retriever(llm, retriever, context_prompt)

    system_prompt = (
        "You are 'PDFSense', a PDF reading and answering assistant. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know. "
        "Answer the questions nicely."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Chain for the chatbot
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_ret, qa_chain)

    # Session ID storing in chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # RAG with history
    conversation_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Enter your question")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversation_rag.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )

        # Display the chat history
        st.write("### Chat History")
        for message in session_history.messages:
            if isinstance(message, dict):  # Handle cases where messages might be dictionaries
                role = message.get("role", "user")  # Default role is 'user'
                content = message.get("content", "")
            else:
                # For LangChain message objects
                role = "user" if isinstance(message, ChatMessageHistory) else "assistant"
                content = message.content

            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.write(content)
            elif role == "system":
                with st.chat_message("system"):
                    st.markdown(f"**System Message:** {content}")

        #st.write("Assistant:", response['answer'])