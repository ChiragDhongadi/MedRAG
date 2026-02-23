# 🏥 MedRAG

### Medical Document Vectorization Pipeline for Retrieval-Augmented Generation

MedRAG is a medical document indexing system that converts PDF-based study materials into a high-performance semantic vector database using FAISS.

The project forms the knowledge foundation for Retrieval-Augmented Generation (RAG) systems by enabling fast, accurate semantic search over medical documents.

---

## 📌 Overview

This repository focuses on:

* Loading medical PDFs from a directory
* Splitting documents into optimized text chunks
* Generating dense vector embeddings
* Building and persisting a FAISS vector store locally

The resulting vector database can be integrated into AI-powered medical summarization, question-answering, or clinical decision-support systems.

---

## 🏗 Architecture

```
Medical PDFs (data/)
        ↓
DirectoryLoader + PyPDFLoader
        ↓
RecursiveCharacterTextSplitter
        ↓
HuggingFace Embeddings (MiniLM)
        ↓
FAISS Vector Store
        ↓
Local Persistence (vectorstore/db_faiss)
```

---

## 🧠 Technology Stack

* Python
* LangChain
* FAISS
* Hugging Face Sentence Transformers
* PyPDF
* all-MiniLM-L6-v2 Embedding Model

---

## 📂 Project Structure

```
MedRAG/
│
├── data/                     # Input medical PDF documents
├── vectorstore/
│   └── db_faiss/             # Generated FAISS index
├── main.py                   # Vector store creation script
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/MedRAG.git
cd MedRAG
```

Install dependencies:

```bash
pip install langchain langchain-community langchain-huggingface \
faiss-cpu sentence-transformers pypdf
```

---

## ▶️ Usage

1. Place your medical PDF files inside the `data/` directory.
2. Run the script:

```bash
python main.py
```

3. The FAISS vector database will be saved to:

```
vectorstore/db_faiss
```

This database can then be loaded into any RAG-based application for semantic retrieval.

---

## 🔧 Configuration

Current parameters:

* Chunk Size: 500 characters
* Chunk Overlap: 50 characters
* Embedding Model: sentence-transformers/all-MiniLM-L6-v2

These can be adjusted based on document size and retrieval performance requirements.

---

## 🎯 Use Cases

* Medical knowledge indexing
* Semantic search over clinical documents
* AI-powered revision systems
* RAG-based medical summarization
* Academic research assistance

---

## 🚀 Roadmap

* Add metadata filtering support
* Support additional document formats (DOCX, images)
* Integrate with LLM for complete RAG pipeline
* Deploy as REST API service
* Add evaluation metrics

---

## 📜 License

This project is intended for educational and research purposes.

