# -*- coding: utf-8 -*-
import os
import sys
import hashlib
import pickle
sys.stdout.reconfigure(encoding='utf-8')

from google import genai


# -----------------------------
# Critique Module Class
# -----------------------------
class CritiqueModule:
    def __init__(self, model_name="gemini-2.5-flash", cache_file="critique_cache.pkl"):
        """
        Initialize the Gemini model client for self-reflective critique.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # -----------------------------
        # Cache setup
        # -----------------------------
        self.cache_file = cache_file
        try:
            with open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)
        except FileNotFoundError:
            self.cache = {}

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
    # Cache key generator
    # -------------------------------------------------
    def _generate_cache_key(self, query: str, docs: list, answer: str = None):
        doc_text = "\n".join([d.page_content for d in docs])
        combined = query + doc_text
        if answer:
            combined += answer
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    # -------------------------------------------------
    # Utility: Interpret Gemini’s critique as Boolean
    # -------------------------------------------------
    def _interpret_response(self, response: str):
        """
        Interprets Gemini's textual response (YES/NO) as a Boolean and extracts feedback.
        """
        response = response.strip()
        lower = response.lower()
        if "yes" in lower:
            return True, response
        elif "no" in lower:
            return False, response
        else:
            # if uncertain, treat as failure
            return False, f"Unclear response from model: {response}"

    # -------------------------------------------------
    # 1️⃣ Check: Retrieved Documents Relevance
    # -------------------------------------------------
    def check_retrieval_relevance(self, query: str, docs: list):
        """
        Check if the retrieved documents are relevant to the query.
        Returns (bool, feedback).
        """
        key = self._generate_cache_key(query, docs)
        if key in self.cache:
            return self.cache[key]

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are a critique model that evaluates the first stage of a Retrieval-Augmented Generation (RAG) pipeline.

TASK:
Determine whether the following retrieved documents contain information that is relevant and helpful for answering the user's query.

Query:
{query}

Retrieved Documents (first 3000 characters):
{context[:3000]}

Respond with 'YES' or 'NO' followed by a short justification.
        """
        response = self._ask_gemini(prompt)
        result = self._interpret_response(response)
        self.cache[key] = result
        self._save_cache()
        return result

    # -------------------------------------------------
    # 2️⃣ Check: Generation Supported by Retrieved Docs
    # -------------------------------------------------
    def check_generation_support(self, answer: str, docs: list):
        """
        Check whether the generated answer is directly supported by the retrieved documents.
        Returns (bool, feedback).
        """
        key = self._generate_cache_key("", docs, answer)
        if key in self.cache:
            return self.cache[key]

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are a critique model checking factual consistency in a RAG system.

TASK:
Determine if the generated answer relies on and aligns with the evidence present in the retrieved documents.

Generated Answer:
{answer}

Retrieved Documents (first 3000 characters):
{context[:3000]}

Respond with 'YES' or 'NO' followed by a brief justification.
        """
        response = self._ask_gemini(prompt)
        result = self._interpret_response(response)
        self.cache[key] = result
        self._save_cache()
        return result

    # -------------------------------------------------
    # 3️⃣ Check: Generation Usefulness for Query
    # -------------------------------------------------
    def check_generation_usefulness(self, query: str, answer: str):
        """
        Evaluate if the generated answer is useful and relevant to the query.
        Returns (bool, feedback).
        """
        key = self._generate_cache_key(query, [], answer)
        if key in self.cache:
            return self.cache[key]

        prompt = f"""
You are a critique model assessing the usefulness of a generated answer
in a RAG system.

Query:
{query}

Generated Answer:
{answer}

TASK:
Determine whether this answer is clear, complete, and useful for the user’s original query.
Respond with 'YES' or 'NO' followed by a concise justification.
        """
        response = self._ask_gemini(prompt)
        result = self._interpret_response(response)
        self.cache[key] = result
        self._save_cache()
        return result

    # -------------------------------------------------
    # 🔄 Query Rephrasing (for self-reflection)
    # -------------------------------------------------
    def rephrase_query(self, query: str, feedback: str):
        """
        Ask Gemini to rephrase the query based on critique feedback.
        Used when relevance or usefulness checks fail.
        """
        prompt = f"""
You are a query rephraser for a RAG system.

Original Query:
{query}

Critique Feedback:
{feedback}

TASK:
Rephrase the query to make it clearer, more specific, and better aligned with the retrieval process.
Only return the improved query text.
        """
        new_query = self._ask_gemini(prompt)
        print(f"🔁 Rephrased Query: {new_query}")
        return new_query

    # -------------------------------------------------
    # Save cache to disk
    # -------------------------------------------------
    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
