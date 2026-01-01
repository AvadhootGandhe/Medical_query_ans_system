import os
from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

faiss_path = r"findyourpath\faiss_db"

vector_store = FAISS.load_local(
    faiss_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")

    results = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in results])

    prompt = f"""
    You are a medical expert. Use the retrieved info to answer safely.

    Retrieved Context:
    {context}

    User Question:
    {query}

    Provide a clear and accurate medical explanation.
    """

    response = llm.invoke(prompt)
    final_answer = response.content

    return jsonify({"answer": final_answer})


@app.route("/", methods=["GET"])
def home():
    return "RAG system running!"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080 )
