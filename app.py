```python
import os
import time
from flask import Flask, request, jsonify

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# CONFIG
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found")

FAISS_PATH = r"findyourpath/faiss_db"

TOP_K_DEFAULT = 5

# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

print("Loading FAISS vector store...")

vector_store = FAISS.load_local(
    FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully")

# ============================================================
# LOAD GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# HELPER FUNCTION
# ============================================================

def build_prompt(context, query):
    return f"""
You are an advanced medical AI assistant.

Answer ONLY using the retrieved medical context below.

If the answer is not present in the context, say:
"I could not find sufficient medical evidence in the retrieved documents."

Do not hallucinate medical facts.

Retrieved Medical Context:
{context}

User Question:
{query}

Provide:
1. A medically accurate explanation
2. Important symptoms if applicable
3. Possible causes if relevant
4. General precautions if relevant

Keep the response clear and professional.
"""

# ============================================================
# MAIN QUERY ENDPOINT
# ============================================================

@app.route("/query", methods=["POST"])
def query():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON body provided"
            }), 400

        query_text = data.get("query")
        top_k = data.get("top_k", TOP_K_DEFAULT)

        if not query_text:
            return jsonify({
                "status": "error",
                "message": "Query field is required"
            }), 400

        # ====================================================
        # RETRIEVAL START
        # ====================================================

        retrieval_start = time.time()

        docs_and_scores = vector_store.similarity_search_with_score(
            query_text,
            k=top_k
        )

        retrieval_end = time.time()

        retrieval_time_ms = round(
            (retrieval_end - retrieval_start) * 1000,
            2
        )

        # ====================================================
        # PROCESS RETRIEVED DOCUMENTS
        # ====================================================

        retrieved_contexts = []

        context_chunks = []

        for rank, (doc, score) in enumerate(docs_and_scores, start=1):

            context_chunks.append(doc.page_content)

            retrieved_contexts.append({
                "rank": rank,
                "similarity_score": round(float(score), 4),
                "source_chunk": doc.page_content[:500]
            })

        context = "\n\n".join(context_chunks)

        # ====================================================
        # PROMPT CREATION
        # ====================================================

        prompt = build_prompt(context, query_text)

        # ====================================================
        # LLM GENERATION
        # ====================================================

        llm_response = llm.invoke(prompt)

        final_answer = llm_response.content

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return jsonify({
            "status": "success",

            "query": query_text,

            "answer": final_answer,

            "retrieved_contexts": retrieved_contexts,

            "retrieval_metrics": {
                "top_k": top_k,
                "retrieval_time_ms": retrieval_time_ms,
                "documents_retrieved": len(retrieved_contexts)
            }
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "vector_db_loaded": True,
        "embedding_model":
            "sentence-transformers/all-mpnet-base-v2",
        "llm_model": "gemini-2.5-flash"
    })


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "MedQuery RAG System Running",
        "status": "online"
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
```
