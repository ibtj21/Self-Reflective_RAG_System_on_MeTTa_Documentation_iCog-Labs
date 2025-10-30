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

        # 🌟 Initialize reasoning trace storage
        reasoning_trace = {
            "original_query": query,
            "iterations": [],
            "final_answer": None,
            "status": None
        }

        while retries <= self.max_retries:
            print(f"\n🔁 Attempt {retries + 1}/{self.max_retries + 1}")

            iteration_log = {"attempt": retries + 1, "query": query}

            # Step 1: Retrieve
            retrieved_docs = self.retriever.retrieve_top_k(query, k=self.k)
            iteration_log["retrieved_docs_preview"] = [
                doc.page_content[:400] for doc in retrieved_docs
            ]

            # Step 2: Critique retrieved documents
            retrieval_ok, retrieval_feedback = self.critic.check_retrieval_relevance(query, retrieved_docs)
            iteration_log["retrieval_critique"] = retrieval_feedback
            iteration_log["retrieval_passed"] = retrieval_ok

            if not retrieval_ok:
                print("⚠️ Retrieval critique failed:", retrieval_feedback)
                query = self.critic.rephrase_query(query, retrieval_feedback)
                iteration_log["rephrased_query"] = query
                reasoning_trace["iterations"].append(iteration_log)
                retries += 1
                continue
            print("✅ Retrieved documents are relevant.")

            # Step 3: Generate
            answer = self.generator.generate_answer(query, retrieved_docs)
            iteration_log["generated_answer"] = answer

            # Step 4: Critique answer
            support_ok, support_feedback = self.critic.check_generation_support(answer, retrieved_docs)
            iteration_log["support_critique"] = support_feedback
            iteration_log["support_passed"] = support_ok

            if not support_ok:
                print("⚠️ Generation not well supported by retrieved docs:", support_feedback)
                reasoning_trace["iterations"].append(iteration_log)
                retries += 1
                continue
            print("✅ Generation is supported by retrieved docs.")

            # Step 5: Check usefulness / final critique
            usefulness_ok, usefulness_feedback = self.critic.check_generation_usefulness(query, answer)
            iteration_log["usefulness_critique"] = usefulness_feedback
            iteration_log["usefulness_passed"] = usefulness_ok

            if not usefulness_ok:
                print("⚠️ Answer not fully useful:", usefulness_feedback)
                query = self.critic.rephrase_query(query, usefulness_feedback)
                iteration_log["rephrased_query"] = query
                reasoning_trace["iterations"].append(iteration_log)
                retries += 1
                continue

            print("✅ Final answer passed all critique checks!")
            print("\n🎯 FINAL ANSWER:")
            print(answer)

            # ✅ Store final result in trace
            reasoning_trace["iterations"].append(iteration_log)
            reasoning_trace["final_answer"] = answer
            reasoning_trace["status"] = "success"

            # Return both answer and trace
            return answer, reasoning_trace

        print("\n❌ Failed to produce a satisfactory answer after several retries.")
        reasoning_trace["status"] = "failed"
        reasoning_trace["final_answer"] = "Failed to generate a high-quality response."
        return "Failed to generate a high-quality response.", reasoning_trace


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    rag_system = SelfReflectiveRAG(max_retries=2, k=3)
    rag_system.retriever.load_vectorstore()

    user_query = "How do I define a function in MeTTa?"

    answer, trace = rag_system.process_query(user_query)
    print("\n🧠 REASONING TRACE:")
    import json
    print(json.dumps(trace, indent=2, ensure_ascii=False))
