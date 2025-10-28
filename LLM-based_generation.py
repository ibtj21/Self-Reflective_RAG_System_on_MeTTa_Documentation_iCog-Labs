v# -*- coding: utf-8 -*-
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
    # Prepare Context from Retrieved Documents
    # -----------------------------
    def prepare_context(self, retrieved_docs):
        """
        Combine the top-k retrieved documents into a single text context.
        """
        if not retrieved_docs:
            return "No retrieved documents available."
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context[:6000]  # truncate for safety (Gemini prompt size)

    # -----------------------------
    # Generate Answer
    # -----------------------------
    def generate_answer(self, query, retrieved_docs):
        """
        Generate a response using Gemini with the query and retrieved document context.
        """
        context = self.prepare_context(retrieved_docs)
        prompt = f"""
You are a helpful AI assistant answering questions based on MeTTa documentation.
Use ONLY the provided context to answer clearly and concisely.

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            answer = response.text.strip()
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            answer = "Error: Failed to generate a response."

        print("\n✅ Query processed successfully!")
        print(f"\nQuestion: {query}\n")
        print(f"Answer: {answer}\n")

        return answer


# -----------------------------
# Example Usage (for testing)
# -----------------------------
if __name__ == "__main__":
    from Document_retrieval import RetrievalEngine  # Import your retrieval class

    retriever = RetrievalEngine()
    generator = LLMGenerator()

    # Ensure the FAISS index is loaded
    retriever.load_vectorstore()

    query = "How do I define a function in MeTTa?"
    top_docs = retriever.retrieve_top_k(query, k=3)
    generator.generate_answer(query, top_docs)
