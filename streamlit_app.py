# -*- coding: utf-8 -*-
import streamlit as st
import requests
import json

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🧠 Self-Reflective RAG System",
    page_icon="📘",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🧠 Self-Reflective RAG System on MeTTa Docs")
st.subheader("Gemini + FAISS + Self-Reflection + Critique")

st.markdown("""
This system answers MeTTa programming questions using **Gemini** with a
**self-reflective Retrieval-Augmented Generation (RAG)** process.  
It also critiques itself and improves its response before finalizing.
""")

# -----------------------------
# Input Section
# -----------------------------
query = st.text_area(
    "💬 Ask a question about MeTTa:",
    placeholder="e.g., How do I define a function in MeTTa?",
    height=100
)

submit = st.button("🚀 Generate Answer")

# -----------------------------
# Backend URL
# -----------------------------
API_URL = "http://127.0.0.1:5000/query"

# -----------------------------
# Response Handling
# -----------------------------
if submit and query.strip():
    with st.spinner("Processing your question through the reflective pipeline... ⏳"):
        try:
            response = requests.post(API_URL, json={"query": query})
            if response.status_code == 200:
                result = response.json()

                st.success("✅ Final Answer Generated!")
                st.markdown(f"### **Question:** {result.get('query', '')}")
                st.markdown(f"### **Answer:**\n\n{result.get('answer', '')}")

                # Reasoning Trace
                if "trace" in result:
                    with st.expander("🧩 Show Full Reasoning Trace"):
                        st.json(result["trace"], expanded=False)
                else:
                    st.info("ℹ️ No reasoning trace available.")
            else:
                st.error(f"❌ Server returned status {response.status_code}")
        except Exception as e:
            st.error(f"⚠️ Error connecting to API: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
✅ **Backend:** Flask API  
✅ **Frontend:** Streamlit  
✅ **LLM:** Gemini  
✅ **Retriever:** FAISS + HuggingFace Embeddings  
""")
