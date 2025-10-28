# -*- coding: utf-8 -*-
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------
# Imports
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Retrieval Engine Class
# -----------------------------
class RetrievalEngine:
    def __init__(self,
                 pdf_folder="Dataset",
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 faiss_index_path="faiss_meetta_index",
                 chunk_size=1000,
                 chunk_overlap=200):
        """
        Initialize the Retrieval Engine with embedding model and FAISS index path.
        """
        self.pdf_folder = pdf_folder
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.faiss_index_path = faiss_index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None # Placeholder for FAISS vector store

    # -----------------------------
    # Load PDF Documents
    # -----------------------------
    def load_documents(self, pdf_paths: list = None):
        documents = []

        # If no specific list is given, load all PDFs in the folder
        if pdf_paths is None:
            pdf_paths = [
                os.path.join(self.pdf_folder, f)
                for f in os.listdir(self.pdf_folder)
                if f.endswith(".pdf")
            ]

        for path in pdf_paths:
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ Loaded {len(docs)} pages from: {os.path.basename(path)}")

        print(f"\n📄 Total pages loaded: {len(documents)} from {len(pdf_paths)} PDF files.\n")
        return documents

    # -----------------------------
    # Split Documents into Chunks
    # -----------------------------
    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len # Character-based splitting (chunks by characters)
        )
        chunks = splitter.split_documents(documents)
        print(f"📄 Split into {len(chunks)} chunks.")
        print(f" Example chunk length: {len(chunks[0].page_content)} characters")
        return chunks

    # -----------------------------
    # Embed and Store Chunks in FAISS
    # -----------------------------
    def embed_and_store(self, chunks):
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.faiss_index_path)
        print(f"💾 FAISS vector store created with {len(chunks)} chunks.")

    # -----------------------------
    # Load FAISS Index from Disk
    # -----------------------------
    def load_vectorstore(self):
        self.vectorstore = FAISS.load_local(
            self.faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True #meaning it unpacks the stored binary data into live Python objects again.
        )                                        #Yes, I understand the risk — go ahead and load this FAISS index anyway.Since its safe when you are the one who created it.

    # -----------------------------
    # Retrieve Top-k Similar Documents
    # -----------------------------
    def retrieve_top_k(self, query, k=3):
        if self.vectorstore is None:
            self.load_vectorstore()
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)
        print(f"🔍 Retrieved top {k} relevant documents for query: '{query}'")
        return retrieved_docs


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Initialize engine
    engine = RetrievalEngine(pdf_folder="Dataset")

    # Load all PDFs from the folder automatically
    docs = engine.load_documents()

    # Split, embed, and store
    chunks = engine.split_documents(docs)
    engine.embed_and_store(chunks)

    # Example query
    query = "How do I define a function in MeTTa?"
    results = engine.retrieve_top_k(query, k=3)

    print("\n--- Retrieved content samples ---")
    for i, doc in enumerate(results):
        print(f"\n[Doc {i+1}]\n{doc.page_content[:500]}...")
