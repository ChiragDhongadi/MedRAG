# 🏥 MedRAG

### Medical Retrieval-Augmented Generation System for Intelligent Revision Notes

MedRAG is an end-to-end Retrieval-Augmented Generation (RAG) system designed to transform medical study materials into structured, high-quality revision sheets.

The system extracts content from PDFs and medical images (via OCR), retrieves relevant domain knowledge using semantic search over a FAISS vector database, and generates personalized revision notes using a Large Language Model.

---

## 🚀 Key Features

* 📄 Multi-document PDF ingestion
* 🖼 OCR-based image text extraction (Tesseract)
* 🔍 Semantic search using FAISS
* 🧠 Retrieval-Augmented Generation (RAG)
* 📝 Personalized learning styles:

  * Bullet points
  * Short notes
  * Detailed notes
  * Example-based explanations
* ⚡ Hallucination-reduction via grounded context
* 🏗 Modular and extensible architecture

---

## 🏗 System Architecture

```
User Upload (PDF / Image)
        ↓
Text Extraction
  - PyPDFLoader
  - Tesseract OCR
        ↓
Text Aggregation
        ↓
Semantic Retrieval (FAISS)
        ↓
Context Construction
        ↓
Prompt Engineering
        ↓
LLM Generation (MedGemma via Ollama)
        ↓
Structured One-Page Revision Sheet
```

---

## 🧠 Tech Stack

* Python
* LangChain
* FAISS
* Hugging Face Embeddings (all-MiniLM-L6-v2)
* Ollama (MedGemma 4B)
* Tesseract OCR
* PIL (Image Processing)

---

## 📂 Project Structure

```
MedRAG/
│
├── uploaded_files/          # User-uploaded PDFs & images
├── vectorstore/
│   └── db_faiss/            # Pre-built FAISS index
├── main.py                  # RAG pipeline script
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/MedRAG.git
cd MedRAG
```

### 2️⃣ Install Dependencies

```bash
pip install langchain langchain-community langchain-huggingface \
faiss-cpu sentence-transformers pypdf pillow pytesseract ollama
```

### 3️⃣ Install Tesseract OCR

* Windows: Install from official Tesseract installer
* Linux:

```bash
sudo apt install tesseract-ocr
```

### 4️⃣ Install Ollama and Pull Model

```bash
ollama pull MedAIBase/MedGemma1.5:4b
```

---

## ▶️ Usage

1. Place medical PDFs or images inside the `uploaded_files/` directory.
2. Ensure your FAISS vector store exists in:

```
vectorstore/db_faiss
```

3. Run:

```bash
python main.py
```

4. Choose your preferred learning style when prompted:

```
bullets / short notes / detailed notes / examples
```

The system will generate a structured one-page revision sheet.

---

## 🔎 How It Works

### 1️⃣ Document Processing

* Loads PDFs via `PyPDFLoader`
* Extracts image text using Tesseract OCR
* Combines all extracted content

### 2️⃣ Semantic Retrieval

* Loads pre-built FAISS vector store
* Retrieves top-k relevant medical knowledge
* Grounds generation in domain-specific context

### 3️⃣ Prompt Engineering

* Enforces strict anti-hallucination rules
* Merges multiple sources
* Adapts output format to user learning preference

### 4️⃣ LLM Generation

* Uses MedGemma 4B via Ollama
* Produces structured revision notes

---

## 🎯 Use Cases

* Medical exam preparation
* Clinical concept revision
* Summarizing large medical PDFs
* Transforming notes into structured revision sheets
* AI-powered study assistant

---

## 🧪 Design Principles

* Context-grounded generation
* Controlled hallucination
* Clean structured outputs
* Modular RAG architecture
* Medical-domain adaptation

---

## 🚀 Future Improvements

* Web UI (Gradio / Streamlit)
* Hugging Face Inference API support
* Token optimization
* Multi-step summarization refinement
* RAG evaluation (RAGAS)
* Dockerized deployment

---

## 🏆 Resume Highlight

> Developed MedRAG, a Retrieval-Augmented Generation system that extracts medical content from PDFs and images, performs semantic search using FAISS, and generates personalized revision sheets using a medical LLM with hallucination control mechanisms.

---
