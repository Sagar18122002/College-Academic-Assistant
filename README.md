# 🎓 College Academic Assistant – RAG-Based Student Helpdesk

A Retrieval-Augmented Generation (RAG) based academic assistant that helps students find answers to college-related questions using academic policy documents.

The application uses **LangChain, Gemini, ChromaDB, Hugging Face Embeddings, Cross-Encoder Reranking, and Streamlit** to retrieve relevant information and generate context-grounded answers.

## 🚀 Project Overview

Students often need to search through multiple documents to find information about attendance requirements, examinations, assignments, library rules, fees, and placement eligibility.

This project provides a centralized AI-powered assistant that retrieves relevant information from academic documents and generates answers using a Large Language Model (LLM).

The system is designed to reduce manual document searching and provide answers with supporting source information.

## 🎯 Objectives

- Build a practical RAG-based application.
- Understand document loading and text splitting.
- Generate embeddings using Hugging Face.
- Implement vector search using ChromaDB.
- Improve retrieval relevance using Cross-Encoder reranking.
- Generate context-grounded responses using Google Gemini.
- Evaluate different RAG approaches.
- Develop a web application using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| LangChain | RAG application framework |
| Google Gemini | Large Language Model |
| Hugging Face | Embedding model |
| Sentence Transformers | Embeddings and reranking |
| Vector similarity search |
| ChromaDB | Vector database |
| Cross-Encoder | Document reranking |
| Streamlit | Web application |
| Pandas | Evaluation and result analysis |
| LangSmith | Tracing and monitoring (planned/integration stage) |

## 🧠 Project Architecture

### V1 – LangChain Basics

Implemented a basic LangChain pipeline:

`Prompt → LLM → Output Parser`


### V2 – ChromaDB-Based RAG

Replaced FAISS with ChromaDB and worked with document metadata such as source, page, and document type.

### V3 – RAG with Reranking

Improved retrieval using a Cross-Encoder:

1. Retrieve the top 10 candidate chunks.
2. Calculate relevance scores using a Cross-Encoder.
3. Rerank the retrieved chunks.
4. Select the top 3 chunks.
5. Pass the selected context to Gemini.
6. Generate a context-grounded answer.

### Streamlit Application

Built a web interface that allows users to:

- Enter academic questions.
- Receive AI-generated answers.
- View retrieved source documents.
- Inspect reranker scores.
- View processing time.

## 📂 Project Structure

```text
College-Academic-Assistant/
│
├── data/
│   └── Academic_Regulations.pdf
│
│
├── codes/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embedding_model.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── llm_service.py
│   └── rag_pipeline.py
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```
## 👨‍💻 Author

Developed as a hands-on Generative AI and Retrieval-Augmented Generation project to explore document retrieval, vector databases, reranking, LLM integration, and AI application development.

**Technologies:** Python | LangChain | RAG | Gemini | FAISS | ChromaDB | Hugging Face | Streamlit
