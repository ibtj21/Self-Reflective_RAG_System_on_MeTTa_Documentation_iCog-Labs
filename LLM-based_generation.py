# -*- coding: utf-8 -*-
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from google import genai

# -----------------------------
# LLM-Based Generation Class
# -----------------------------
class LLMGenerator:
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the Gemini model client for answer generation.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # -----------------------------
    # Format the context from retrieved documents
    # -----------------------------
    def prepare_context(self, retrieved_docs):
        """
        Combine the top-k retrieved documents into a single text context.
        """
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context

    # -----------------------------
    # Generate Answer
    # -----------------------------
    def generate_answer(self, query, retrieved_docs):
        """
        Generate a response from Gemini using the query and retrieved documents.
        """
        context = self.prepare_context(retrieved_docs)

        # Make a generation request
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=f"Context:\n{context}\n\nQuestion: {query}"
        )

        print("\n✅ Query processed successfully!")
        print("\nQuestion:", query)
        print("\nAnswer:", response.text)
        return response.text


# -----------------------------
# Example Usage (for testing)
# -----------------------------
if __name__ == "__main__":
    from Document_retrieval import RetrievalEngine  # Import your retrieval class

    # Step 1: Initialize retriever and generator
    retriever = RetrievalEngine()
    generator = LLMGenerator()

    # Step 2: Retrieve top-k docs
    query = "How do I define a function in MeTTa?"
    retriever.load_vectorstore()
    top_docs = retriever.retrieve_top_k(query, k=3)

    # Step 3: Generate answer using LLM
    generator.generate_answer(query, top_docs)
