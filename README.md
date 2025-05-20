# Intelligent Trading Q&A System

A web-based application that allows users to ask questions about trading and receive contextually relevant answers. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks and LangChain to integrate retrieval with a large language model (LLM) for answer generation. A Streamlit interface provides a user-friendly front end.

This project demonstrates expertise in natural language processing (NLP), vector databases, and LLM-powered applications, making it ideal for portfolio showcasing.

## Features
- Extracts text from PDF technical manuals for processing.
- Uses RAG to combine document retrieval with LLM-based answer generation.
- Built with LangChain for orchestrating the RAG pipeline.
- Stores document embeddings in a FAISS vector database for fast similarity search.
- Interactive Streamlit web interface for querying manuals.
- Custom prompt engineering to ensure clean, accurate answers.

## Demo
https://app-app-6esvqcxwsuspb5yshvzhva.streamlit.app/

## Project Structure
```bash
rag-langchain/
├── data/                       # Folder for input/output data                 
├── scripts/                    # Source code for loading, processing, querying
│   ├── chunk_documents.py
│   ├── extract_text.py
│   ├── rag_pipeline.py
│   └── create_vector_store.py
├── app.py                      # Streamlit or main script
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
