# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify
from SelfReflectiveRAG import SelfReflectiveRAG

# -----------------------------
# Initialize Flask app and RAG system
# -----------------------------
app = Flask(__name__)

rag = SelfReflectiveRAG(max_retries=2, k=3)
rag.retriever.load_vectorstore()

# -----------------------------
# Define the API route
# -----------------------------
@app.route("/query", methods=["POST"])
def query_rag():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400

    user_query = data["query"]
    print(f"\n🧠 Received query: {user_query}")

    answer = rag.process_query(user_query)
    return jsonify({"query": user_query, "answer": answer})


# -----------------------------
# Run the Flask app
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Self-Reflective RAG API Server on http://127.0.0.1:5000/query")
    app.run(debug=True)
