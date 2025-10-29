# -*- coding: utf-8 -*-
import streamlit as st
import requests
import json

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Self-Reflective RAG on MeTTa Docs",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🧠 Self-Reflective RAG System")
st.subheader("Built on MeTTa Documentation using Gemini + FAISS")

st.markdown("""
This system retrieves relevant sections from the MeTTa documentation,
generates an answer using **Gemini**, and then critiques its own reasoning
to ensure clarity, factual consistency, and usefulness.
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
API_URL = "http://127.0.0.1:5000/query"  # Flask must be running!

# -----------------------------
# Response Section
# -----------------------------
if submit and query.strip():
    with st.spinner("Processing your query with the Self-Reflective RAG pipeline... ⏳"):
        try:
            response = requests.post(API_URL, json={"query": query})
            if response.status_code == 200:
                result = response.json()

                st.success("✅ Answer Generated Successfully!")
                st.markdown(f"### **Question:** {result.get('query', '')}")
                st.markdown(f"### **Answer:**\n\n{result.get('answer', '')}")

                # Display more info if available (like reflections or retries)
                if "reflection_steps" in result:
                    with st.expander("🪞 Reflection Details"):
                        st.json(result["reflection_steps"], expanded=False)
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
