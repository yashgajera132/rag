import os
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import BaseRotatingHandler
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Add root directory to sys.path so config can be imported regardless of execution context
try:
    import config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

from smart_bot.rag import ask_rag, clear_rag, load_or_build_index, get_memory_contents

app = Flask(__name__)
CORS(app)  # Allow Streamlit to make requests to this server

# ---------------------------------------------------------------------------
# Daily Rotating File Handler — creates logs/YYYY-MM-DD.log
# ---------------------------------------------------------------------------
class DailyFileHandler(BaseRotatingHandler):
    """A log handler that writes to a new file every calendar day."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._current_date = self._today()
        filename = self._log_path()
        super().__init__(filename, mode="a", encoding="utf-8", delay=False)

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _log_path(self) -> str:
        return os.path.join(self.log_dir, f"{self._current_date}.log")

    def shouldRollover(self, record) -> bool:
        return self._today() != self._current_date

    def doRollover(self):
        """Switch the handler to the new day's log file."""
        if self.stream:
            self.stream.close()
            self.stream = None
        self._current_date = self._today()
        self.baseFilename = os.path.abspath(self._log_path())
        self.stream = self._open()


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

# Console handler (for HTTP logs on console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Daily file handler (for all logs in file)
daily_handler = DailyFileHandler(config.LOGS_DIR)
daily_handler.setFormatter(formatter)

# Parent "smart_bot" logger — captures all logs in the daily handler
logger = logging.getLogger("smart_bot")
logger.setLevel(logging.INFO)
logger.addHandler(daily_handler)

# Child "smart_bot.http" logger — handles HTTP requests (logs propagate to parent handler)
http_logger = logging.getLogger("smart_bot.http")
http_logger.setLevel(logging.INFO)
http_logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Configuration & State
# ---------------------------------------------------------------------------
HARDCODED_PDF = config.DEFAULT_PDF
chat_history = []
pdf_status = {"uploaded": True, "filename": HARDCODED_PDF, "index_info": "Loaded from local directory"}

UPLOAD_FOLDER = config.UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load existing index on startup; if none exists, build it
load_or_build_index(HARDCODED_PDF)
logger.info("=" * 60)
logger.info("SERVER STARTED | Smart PDF Q&A Bot | http://%s:%s", config.FLASK_HOST, config.FLASK_PORT)
logger.info("Active PDF: %s", HARDCODED_PDF)
logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.before_request
def log_request_start():
    """Log every incoming HTTP request and start a timer."""
    g.start_time = time.time()
    http_logger.info(
        "REQUEST  | %s %s | IP: %s | Body: %s",
        request.method,
        request.path,
        request.remote_addr,
        request.content_length or 0,
    )


@app.after_request
def log_request_end(response):
    """Log the response status and how long the request took."""
    duration = time.time() - g.get("start_time", time.time())
    http_logger.info(
        "RESPONSE | %s %s | Status: %s | Duration: %.2fs",
        request.method,
        request.path,
        response.status_code,
        duration,
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """Upload a PDF dynamically and rebuild the RAG index."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        # Save file to uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        logger.info("File saved to: %s", file_path)

        # Force rebuild index from the new PDF
        success = load_or_build_index(file_path, force_rebuild=True)
        if not success:
            return jsonify({"error": "Failed to index the uploaded PDF"}), 500

        # Update pdf status and clear standard history logs
        global pdf_status, chat_history
        pdf_status = {
            "uploaded": True,
            "filename": file.filename,
            "index_info": f"Indexed {file.filename} dynamically"
        }
        chat_history.clear()

        logger.info("PDF '%s' uploaded and indexed successfully", file.filename)
        return jsonify({
            "success": True,
            "filename": file.filename,
            "message": f"Successfully uploaded and indexed '{file.filename}'!"
        })

    except Exception as e:
        logger.error("Upload error: %s", str(e))
        return jsonify({"error": f"Upload error: {str(e)}"}), 500


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """Ask a question to the RAG chain."""
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]

    try:
        logger.info("Question: %s", question)

        # Call RAG pipeline — now includes token_usage
        result = ask_rag(question)

        answer = result["answer"]
        structured_items = result.get("structured_items", [])
        sources = result.get("sources", [])
        token_usage = result.get("token_usage", {})
        validation = result.get("validation", {})

        # Store in standard chat history logs
        chat_history.append({"role": "user", "content": question})
        chat_history.append({
            "role": "assistant",
            "content": answer,
            "structured_items": structured_items,
            "sources": sources,
            "validation": validation,
            "token_usage": token_usage
        })

        logger.info("Memory Window: %s", get_memory_contents())
        logger.info("Answer generated successfully (Validation: %s)", validation)

        return jsonify({
            "success": True,
            "answer": answer,
            "structured_items": structured_items,
            "sources": sources,
            "token_usage": token_usage,
            "validation": validation,
        })

    except Exception as e:
        logger.error("RAG error: %s", str(e))
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
    logger.info("Chat history and memory cleared")
    return jsonify({"success": True, "message": "Chat history cleared!"})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" Smart PDF Q&A Bot — Flask Backend")
    print(f" Running on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("=" * 50 + "\n")

    app.run(debug=True, host=config.FLASK_HOST, port=config.FLASK_PORT)
