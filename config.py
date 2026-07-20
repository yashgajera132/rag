import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# # --- API Keys & Credentials ---
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Configurations ---
# LLM Model (Groq)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# Embedding Model (HuggingFace)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_NORMALIZE = False

# Reranker Model (Cross-Encoder)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Retrieval & Memory Parameters ---
RETRIEVER_K = 8
RERANK_TOP_K = 3
ENSEMBLE_WEIGHTS = [0.5, 0.5]
MEMORY_WINDOW_SIZE = 3

# --- Storage & Path Configurations ---
FAISS_INDEX_DIR = "faiss_index_local"
DEFAULT_PDF = "sample-20-page-pdf-a4-size.pdf"
UPLOAD_FOLDER = "uploads"

# --- Server Configurations ---
FLASK_HOST = "localhost"
FLASK_PORT = 5000
FLASK_URL = f"http://{FLASK_HOST}:{FLASK_PORT}"
