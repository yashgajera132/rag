import os
import logging
import time
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from smart_bot.rag import ask_rag, clear_rag, load_or_build_index, get_memory_contents

app = Flask(__name__)
CORS(app)  # Allow Streamlit to make requests to this server

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smart_bot")


@app.before_request
def log_request_start():
    """Log every incoming HTTP request and start a timer."""
    g.start_time = time.time()
    logger.info(
        "➡️  REQUEST  | %s %s | IP: %s | Body: %s",
        request.method,
        request.path,
        request.remote_addr,
        request.content_length or 0,
    )


@app.after_request
def log_request_end(response):
    """Log the response status and how long the request took."""
    duration = time.time() - g.get("start_time", time.time())
    logger.info(
        "⬅️  RESPONSE | %s %s | Status: %s | Duration: %.2fs",
        request.method,
        request.path,
        response.status_code,
        duration,
    )
    return response


# ---------- Configuration & State ----------
HARDCODED_PDF = "sample-20-page-pdf-a4-size.pdf"
chat_history = []
pdf_status = {"uploaded": True, "filename": HARDCODED_PDF, "index_info": "Loaded from local directory"}


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """Ask a question to the RAG chain."""
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]

    try:
        logger.info("❓ Question: %s", question)
        result = ask_rag(question)

        # Store in standard chat history logs
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": result["answer"]})

        logger.info("🧠 Current Memory Window: %s", get_memory_contents())
        logger.info("💡 Answer generated successfully")

        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
        })

    except Exception as e:
        logger.error("❌ RAG error: %s", str(e))
        return jsonify({"error": f"RAG error: {str(e)}"}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get the conversation history and memory contents."""
    return jsonify({
        "history": chat_history,
        "memory_window": get_memory_contents(),
        "pdf_status": pdf_status,
    })


@app.route("/api/clear", methods=["POST"])
def clear_all():
    """Clear chat history and RAG memory window."""
    chat_history.clear()
    clear_rag()
    logger.info("🗑️ Chat history and memory cleared")
    return jsonify({"success": True, "message": "Chat history cleared!"})


if __name__ == "__main__":
    # Try to load existing index on startup; if none exists, build it
    load_or_build_index(HARDCODED_PDF)

    print("\n" + "=" * 50)
    print(" Smart PDF Q&A Bot — Flask Backend")
    print(" Running on http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5000)
