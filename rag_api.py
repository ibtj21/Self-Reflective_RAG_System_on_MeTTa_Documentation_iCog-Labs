# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify
from SelfReflectiveRAG import SelfReflectiveRAG
import os
import json
from datetime import datetime

# -----------------------------
# Initialize Flask app and RAG system
# -----------------------------
app = Flask(__name__)

rag = SelfReflectiveRAG(max_retries=2, k=3)
rag.retriever.load_vectorstore()

# Create folder to save reasoning traces if not exists
TRACE_DIR = "reasoning_traces"
os.makedirs(TRACE_DIR, exist_ok=True)

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

    # Process query — expecting (final_answer, reasoning_trace)
    result = rag.process_query(user_query)

    # Handle tuple or legacy single return
    if isinstance(result, tuple) and len(result) == 2:
        answer, reasoning_trace = result
    else:
        answer = result
        reasoning_trace = getattr(rag, "last_reasoning_trace", None)

    # Save reasoning trace (if available)
    if reasoning_trace:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trace_{timestamp}.json"
        filepath = os.path.join(TRACE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(reasoning_trace, f, ensure_ascii=False, indent=4)

    # Build clean response
    response = {
        "query": user_query,
        "answer": answer if isinstance(answer, str) else reasoning_trace.get("final_answer", "")
    }

    if reasoning_trace:
        response["trace"] = reasoning_trace

    return jsonify(response)


# -----------------------------
# Run the Flask app
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Self-Reflective RAG API Server on http://127.0.0.1:5000/query")
    app.run(debug=True)
