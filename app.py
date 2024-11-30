# Importing libraries
import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# Embeddings and LLM initialization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="Gemma2-9b-It")
st.set_page_config(page_title="PDFSense", page_icon="📜")
# Streamlit app title
st.title("📜 𝐏𝐃𝐅𝐒𝐞𝐧𝐬𝐞 : 𝐏𝐃𝐅 𝐐𝐮𝐞𝐬𝐭𝐢𝐨𝐧 𝐀𝐧𝐬𝐰𝐞𝐫𝐢𝐧𝐠 𝐚𝐬𝐬𝐢𝐬𝐭𝐚𝐧𝐭 𝐰𝐢𝐭𝐡 𝐂𝐡𝐚𝐭 𝐇𝐢𝐬𝐭𝐨𝐫𝐲")

# PDF Uploader Section (Keeps it at the top)
uploaded_files = st.file_uploader("Drop PDF files here", type="pdf", accept_multiple_files=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I am PDFSense. Upload your PDF and ask me anything related to it."}
    ]

# Process PDFs if uploaded
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = "./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        docs = PyPDFLoader(temppdf).load()
        documents.extend(docs)
    os.remove("./temp.pdf")  # Clean up temporary file

    # Text splitting and FAISS index creation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    faiss_index = FAISS.from_documents(splits, embeddings)
    retriever = faiss_index.as_retriever()

    # History-aware retriever and prompt setup
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Refactor the question using chat history for context."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_ret = create_history_aware_retriever(llm, retriever, context_prompt)

    system_prompt = (
        "You are PDFSense, a PDF reading assistant. Use the following context to answer the question: "
        "{context}. If unsure, respond with 'I don't know.'"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_ret, qa_chain)

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User input handling
if user_input := st.chat_input(placeholder="Ask a question about your uploaded PDF..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Run retrieval and answer generation using invoke()
    with st.chat_message("assistant"):
        chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"]]
        result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        
        # Extract and display only the answer
        answer = result.get("answer", "I don't know.")
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.write(answer)