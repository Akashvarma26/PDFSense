# 📜 PDFSense : PDF Question Answering Assistant with Chat History           

PDFSense is an LLM-powered Streamlit application that enables users to upload PDFs and ask questions based on the document's content. It uses a Retrieval-Augmented Generation (RAG) approach to provide accurate, context-aware answers by incorporating previous chat history of the current session.               

[App in Hugging Face Space](https://huggingface.co/spaces/AkashVD26/pdfsense)

## 🚀 Features
- Upload and analyze PDF documents.
- Ask questions about the uploaded PDF in natural language.
- Retrieve answers using LangChain, FAISS indexing, and Hugging Face embeddings.
- Maintain conversation context for coherent responses.    

## 📚 How It Works
- Upload PDF: Drag and drop your PDF file into the uploader.
- Ask Questions: Type a question about the PDF's content.
- Contextual Answers: PDFSense retrieves answers using FAISS and LLMs while maintaining chat history for context.

## 🛠️ Technologies Used         
- Streamlit: Interactive web application framework.
- LangChain: Framework for creating LLM-based applications.
- FAISS: Vector search for efficient retrieval.
- Hugging Face: Pretrained embeddings for document processing.
- Groq: LLM used for generating responses.
- PyPDFLoader: Document loader for processing PDFs.             

## 🧩 Prerequisites
Make sure you have the following prerequisites:

- [Python 3.8 and above](https://www.python.org)
- [Hugging Face account](https://huggingface.co)
- [Hugging Face Access Token](https://huggingface.co/settings/tokens)
- [Groq API key](https://console.groq.com/keys)

## 📦 Installation
If you want to use this locally on your system:

```
git clone https://github.com/Akashvarma26/PDFSense.git
```

```
pip install -r requirements.txt
```

## ▶️ Usage
Run the Streamlit app locally:
```
streamlit run app.py
```

## 🙋‍♂️ Acknowledgments
- [LangChain](https://www.langchain.com)
- [Hugging Face](https://huggingface.co)
- [FAISS](https://ai.meta.com/tools/faiss/)
- [Groq](https://groq.com)
- [streamlit](https://www.langchain.com)

## Configuration for HF Space
---
title: Pdfsense
emoji: 📜
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: PDF Answering Assistant
---