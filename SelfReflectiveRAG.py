# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from Document_retrieval import RetrievalEngine
from LLM_based_generation import LLMGenerator
from Critique_module import CritiqueModule


class SelfReflectiveRAG:
    def __init__(self, max_retries=2, k=3):
        """
        Orchestrates the Retrieval-Augmented Generation (RAG) pipeline
        with a self-reflection critique mechanism.
        """
        self.retriever = RetrievalEngine()
        self.generator = LLMGenerator()
        self.critic = CritiqueModule()
        self.max_retries = max_retries
        self.k = k

    # -----------------------------
    # Main process
    # -----------------------------
    def process_query(self, query):
        print(f"\n🚀 Starting Self-Reflective RAG for query:\n'{query}'\n")
        retries = 0

        while retries <= self.max_retries:
            print(f"\n🔁 Attempt {retries + 1}/{self.max_retries + 1}")

            # Step 1: Retrieve
            retrieved_docs = self.retriever.retrieve_top_k(query, k=self.k)

            # Step 2: Critique retrieved documents
            retrieval_ok, retrieval_feedback = self.critic.check_retrieval_relevance(query, retrieved_docs)
            if not retrieval_ok:
                print("⚠️ Retrieval critique failed:", retrieval_feedback)
                query = self.critic.rephrase_query(query, retrieval_feedback)
                retries += 1
                continue
            print("✅ Retrieved documents are relevant.")

            # Step 3: Generate
            answer = self.generator.generate_answer(query, retrieved_docs)

            # Step 4: Critique answer
            support_ok, support_feedback = self.critic.check_generation_support(answer, retrieved_docs)
            if not support_ok:
                print("⚠️ Generation not well supported by retrieved docs:", support_feedback)
                retries += 1
                continue
            print("✅ Generation is supported by retrieved docs.")

            # Step 5: Check usefulness / final critique
            usefulness_ok, usefulness_feedback = self.critic.check_generation_usefulness(query, answer)
            if not usefulness_ok:
                print("⚠️ Answer not fully useful:", usefulness_feedback)
                query = self.critic.rephrase_query(query, usefulness_feedback)
                retries += 1
                continue
            print("✅ Final answer passed all critique checks!")

            print("\n🎯 FINAL ANSWER:")
            print(answer)
            return answer

        print("\n❌ Failed to produce a satisfactory answer after several retries.")
        return "Failed to generate a high-quality response."


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    rag_system = SelfReflectiveRAG(max_retries=2, k=3)
    rag_system.retriever.load_vectorstore()

    user_query = "Explain how pattern matching works in MeTTa with examples."


    rag_system.process_query(user_query)
