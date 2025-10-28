# -*- coding: utf-8 -*-
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from google import genai


# -----------------------------
# Critique Module Class
# -----------------------------
class CritiqueModule:
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the Gemini model client for self-reflective critique.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # -------------------------------------------------
    # Helper: Send critique prompt to Gemini
    # -------------------------------------------------
    def _ask_gemini(self, prompt: str) -> str:
        """
        Sends a critique prompt to Gemini and returns its textual response.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini Critique Error: {e}")
            return "Error"

    # -------------------------------------------------
    # 1️⃣ Check: Retrieved Documents Relevance
    # -------------------------------------------------
    def check_docs_relevance(self, query: str, docs: list) -> str:
        """
        Check if the retrieved documents are relevant to the query.
        Returns 'YES' or 'NO' with justification.
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are a critique model that evaluates the first stage of a Retrieval-Augmented Generation (RAG) pipeline.

TASK:
Determine whether the following retrieved documents are relevant for answering the given query.

Query:
{query}

Retrieved Documents (first 3000 characters):
{context[:3000]}

Respond with 'YES' or 'NO' followed by a short justification.
        """
        return self._ask_gemini(prompt)

    # -------------------------------------------------
    # 2️⃣ Check: Generation Supported by Retrieved Docs
    # -------------------------------------------------
    def check_generation_supported(self, answer: str, docs: list) -> str:
        """
        Check whether the generated answer is directly supported
        by the retrieved documents.
        Returns 'YES' or 'NO' with justification.
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are a critique model checking factual consistency in a RAG system.

TASK:
Determine if the generated answer is supported by the retrieved documents.

Generated Answer:
{answer}

Retrieved Documents (first 3000 characters):
{context[:3000]}

Respond with 'YES' or 'NO' followed by a brief justification.
        """
        return self._ask_gemini(prompt)

    # -------------------------------------------------
    # 3️⃣ Check: Generation Usefulness for Query
    # -------------------------------------------------
    def check_usefulness(self, query: str, answer: str) -> str:
        """
        Evaluate if the generated answer is useful and relevant to the query.
        Returns 'YES' or 'NO' with justification.
        """
        prompt = f"""
You are a critique model assessing the usefulness of a generated answer
in a RAG system.

Query:
{query}

Generated Answer:
{answer}

TASK:
Determine whether this answer is clear, correct, and directly useful for the query.
Respond with 'YES' or 'NO' followed by a concise justification.
        """
        return self._ask_gemini(prompt)
