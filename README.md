# 🧠 Self-Reflective RAG System on MeTTa Documentation

## 📘 Overview
This project implements a **Self-Reflective Retrieval-Augmented Generation (RAG)** system built on **MeTTa standard library documentation**.  
It integrates **retrieval**, **LLM-based generation**, and a **self-critique module** that reflects on its own answers to iteratively improve response quality.

The system uses:
- **FAISS** for efficient document retrieval  
- **Gemini (Google’s LLM)** for answer generation  
- **Custom Critique Module** for self-evaluation  
- **Flask + Streamlit** for a minimal, interactive user interface

---

## 🎯 Objective
To design and implement a **modular RAG backend** that not only retrieves and generates answers but also **self-evaluates** them across three dimensions:

1. **Retrieved Documents Relevance**  
2. **Generation Supported by Retrieved Docs**  
3. **Generation Usefulness**

---

## 🧩 System Architecture
              ┌──────────────────────┐
              │      User Query      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Retrieval Engine    │
              │ (FAISS + Embeddings) │
              └──────────┬───────────┘
                         │ Retrieved Docs
                         ▼
              ┌──────────────────────┐
              │   LLM Generator      │
              │ (Gemini Model)       │
              └──────────┬───────────┘
                         │ Draft Answer
                         ▼
              ┌──────────────────────┐
              │   Critique Module    │
              │ (Self-Reflection)    │
              ├──────────────────────┤
              │  1. Retrieval Check  │
              │  2. Support Check    │
              │  3. Usefulness Check │
              └──────────┬───────────┘
                         │
                  Retry / Improve
                         ▼
              ┌──────────────────────┐
              │   Final Answer       │
              └──────────────────────┘

---

## ⚙️ Components

### 1. **Document Retrieval Engine** (`Document_retrieval.py`)
- Loads **MeTTa documentation PDFs**
- Splits text into chunks using `RecursiveCharacterTextSplitter`
- Generates vector embeddings via **HuggingFace** models (`all-MiniLM-L6-v2`)
- Stores and retrieves document vectors using **FAISS**

### 2. **LLM-Based Generation** (`LLM_based_generation.py`)
- Uses **Gemini** (`gemini-2.5-flash`) via Google’s GenAI API  
- Generates structured answers grounded in retrieved documentation  
- Ensures responses are **context-aware**, **detailed**, and **code-inclusive**

### 3. **Critique Module** (`Critique_module.py`)
- Performs **three levels of reflection**:
  - **Relevance Check:** Are retrieved documents relevant to the query?  
  - **Support Check:** Is the generated answer supported by retrieved content?  
  - **Usefulness Check:** Is the final answer useful and clear to the user?  
- Rephrases queries or requests re-generation if quality checks fail

### 4. **Self-Reflective RAG Orchestrator** (`SelfReflectiveRAG.py`)
- Coordinates the **retrieval → generation → critique → reflection** loop  
- Maintains a **reasoning trace** for transparency and debugging  
- Retries automatically until a satisfactory answer is reached

### 5. **Flask Backend API** (`app.py`)
- Exposes a `/query` endpoint for the frontend  
- Receives user queries, processes them via the RAG pipeline, and returns JSON responses  
- Saves reasoning traces as JSON files in `reasoning_traces/`

### 6. **Streamlit Frontend** (`app_streamlit.py`)
- Provides a **simple, elegant interface** for interacting with the backend  
- Displays the **final answer**, **question**, and optional **reasoning trace** for inspection

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/self-reflective-rag-metta.git
cd self-reflective-rag-metta
```
### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Set your Gemini API key
```bash
export GOOGLE_API_KEY="your_api_key_here"  # Linux/Mac
set GOOGLE_API_KEY="your_api_key_here"     # Windows
```
### 5. Prepare dataset
```bash
Place MeTTa documentation PDF files in the Dataset/ folder.
```
### 6. Build FAISS index
```bash
python Document_retrieval.py
```

## Running the System
```bash
streamlit run app_streamlit.py
http://localhost:8501
```

### Key Features

✅ Modular design for easy component replacement
✅ Automated self-critique and query rephrasing
✅ Supports any open LLM (currently Gemini)
✅ Reasoning trace for transparency
✅ Minimal Streamlit UI for demo purposes




