# 🧠 Intelligent Complaint Analysis for Financial Services

This project implements an intelligent pipeline to analyze, clean, embed, and retrieve consumer complaints (e.g. CFPB dataset). It leverages NLP techniques, semantic embeddings, and vector stores to support Retrieval-Augmented Generation (RAG) and classification tasks.

---

## 🚀 Project Objectives

- Preprocess and clean financial service complaints.
- Perform Exploratory Data Analysis (EDA).
- Split long narratives into semantically coherent chunks.
- Embed chunks using Sentence Transformers.
- Store embeddings in a vector database (ChromaDB or FAISS).
- Enable efficient semantic search and RAG-based applications.

---

## 📁 Project Structure

├── data/
│ ├── complaints.csv
│ ├── processed/
│ │ └── filtered_complaints.csv
│ └── vector_store/
├── notebooks/
│ └── task1_eda_preprocessing.ipynb
├── src/
│ ├── task2_embedding_indexing.py
│ └── ...
├── outputs/
│ └── *.png (EDA plots)
├── requirements.txt
└── README.md


---

## 🛠️ Tasks Overview

### ✅ Task 1: EDA & Preprocessing
- Load full CFPB complaint dataset.
- Visualize distribution across product types.
- Analyze narrative length statistics.
- Filter to 5 core products:
  - `Credit card`, `Personal loan`, `Buy Now, Pay Later`, `Savings account`, `Money transfers`
- Remove missing/empty narratives.
- Normalize and clean complaint text.

📄 Script/Notebook: `notebooks/task1_eda_preprocessing.ipynb`  
📦 Output: `data/processed/filtered_complaints.csv`

---

### ✅ Task 2: Text Chunking, Embedding, Indexing
- Split long complaint texts into overlapping 500-character chunks.
- Generate sentence embeddings using:

- Store vectors and metadata (complaint ID, product type) in a ChromaDB vector store.

📄 Script: `src/task2_embedding_indexing.py`  
📦 Output: `data/vector_store/`

---

### ✅ Task 3 (Optional): RAG & Search (Upcoming)
- Retrieve relevant chunks for a user query using semantic similarity.
- Build interface with FastAPI or Gradio (optional).

📄 Script: `src/task3_rag_query.py` *(optional)*  
📦 Output: relevant top-k chunks, source metadata

---

## 🧪 Environment Setup

### 🧰 Dependencies

Install required packages:
```bash
pip install -r requirements.txt
