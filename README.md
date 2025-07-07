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

📄 Script: `notebooks/RAG_pipeline.ipynb`  
📦 Output: `data/vector_store/`

---

### 🧠 Task 3: RAG Inference (CPU Compatible)

- Use retrieved complaint chunks as context
- Generate answers using `google/flan-t5-base` (lightweight and CPU-friendly)
- Format queries and responses appropriately

📄 Script: `notebooks/vector_store_indexing.ipynb`  
📦 Output: relevant top-k chunks, source metadata


### 💬 Task 4: Streamlit Chat Interface

- Web interface for interacting with the RAG system
- Ask financial questions and receive AI-generated answers
- Show source complaint chunks below answers for transparency
- Adjustable `top_k` chunk retrieval via sidebar
- "Clear" button to reset session

📄 Script: `src/app.py`



## 🧪 Environment Setup

### 🧰 Dependencies

Install required packages:
```bash
pip install -r requirements.txt

🖥️ How to run:
```bash
streamlit run app.py

---