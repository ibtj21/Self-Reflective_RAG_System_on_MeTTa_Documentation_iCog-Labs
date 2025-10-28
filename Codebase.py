# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Step 1: Import required loader
from langchain_community.document_loaders import PyPDFLoader

# Step 2: Define the path to your PDF
pdf_path = "MeTTa Standard Library Documentation — MeTTa Standard Library 0.1 documentation.pdf"

# Step 3: Initialize the loader
loader = PyPDFLoader(pdf_path)

# Step 4: Load the document
documents = loader.load()

# Step 5: Inspect summary (avoid printing full content)

print(f"Loaded {len(documents)} pages from: {pdf_path}")

# Optional: preview first page
print("\n--- Sample from first page ---")
#print(documents[0].page_content[:500])


# Optional: print metadata of the first page
print("\n--- Metadata of first page ---")       
#print(documents[0].metadata)


from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split the loaded documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # max characters per chunk
    chunk_overlap=200, # overlap between chunks to preserve context
    length_function=len
)

chunks = text_splitter.split_documents(documents)

print(f" Split into {len(chunks)} chunks.")
print(f" Example chunk length: {len(chunks[0].page_content)} characters")

# Optional: preview a small portion of the first chunk
print("\n--- Sample from first chunk ---")
#print(chunks[0].page_content[:500])
# Optional: print lengths of all chunks
print("\n--- Lengths of all chunks ---")
print([len(chunk.page_content) for chunk in chunks])

# -*- coding: utf-8 -*-
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai  # Official Google Gemini client

# -----------------------------
# Embedding & Storing
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_meetta_index")
print(f"FAISS vector store created with {len(chunks)} chunks.")

# -----------------------------
# Load stored FAISS index
# -----------------------------
vectorstore = FAISS.load_local("faiss_meetta_index", embeddings, allow_dangerous_deserialization=True)

# -----------------------------
# Initialize Gemini client
# -----------------------------
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# Query + Retrieval
# -----------------------------
query = "How do I define a function in MeTTa?"
retrieved_docs = vectorstore.similarity_search(query, k=3)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# -----------------------------
# Generate answer using Gemini
# -----------------------------
response = client.models.generate_content(
    model="gemini-2.5-flash",  # replace with your valid model from AI Studio
    contents=f"Context:\n{context}\n\nQuestion: {query}"
)

print("\nQuestion:", query)
print("\nAnswer:", response.text)
