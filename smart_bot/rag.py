import os
import sys
import logging
import time
import math
from pathlib import Path
from typing import List, Dict, Any

# Add root directory to sys.path so config can be imported regardless of execution context
try:
    import config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import CrossEncoder
from rapidfuzz import fuzz

logger = logging.getLogger("smart_bot")

# Pydantic schema that the LLM must output as JSON
class LLMResponseSchema(BaseModel):
    answer: str = Field(
        description=(
            "A concise text answer to the user's question. "
            "If the question asks for a list or definitions, write a brief summary here."
        )
    )
    structured_items: List[Dict[str, Any]] = Field(
        default=[],
        description=(
            "A list of structured items when the user asks for a list, glossary, table, or definitions. "
            "Each item is a dict, e.g. {\"term\": \"Annotation\", \"definition\": \"A comment...\"}. "
            "Leave as an empty list [] for simple factual questions."
        )
    )


# Embedding model shared for index building and retrieval
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": config.EMBEDDING_NORMALIZE}
)

vector_store = None
bm25_retriever = None

# Initialize Cross-Encoder reranker globally
reranker = CrossEncoder(config.RERANKER_MODEL_NAME)

# Sliding window memory keeping the last 3 QA turns
memory = ConversationBufferWindowMemory(
    k=config.MEMORY_WINDOW_SIZE,
    memory_key="chat_history",
    return_messages=True
)


def validate_answer(answer: str, chunk_texts: List[str], threshold: float) -> tuple:
    """
    Validate the generated answer against the retrieved chunk texts using fuzzy matching.
    Returns (is_valid, max_score).
    """
    if not answer or answer.strip() in [
        "I don't know based on the provided document.",
        "No PDF has been uploaded/indexed yet."
    ]:
        return True, 100.0

    try:
        # Calculate token_set_ratio similarities against each chunk text
        scores = [fuzz.token_set_ratio(answer, chunk) for chunk in chunk_texts]
        max_score = max(scores) if scores else 0.0

        return max_score >= threshold, max_score
    except Exception as e:
        logger.error("Error in validate_answer: %s", e)
        return True, 100.0  # Fallback to True to avoid infinite loops if something crashes


def load_or_build_index(pdf_path: str, save_dir: str = config.FAISS_INDEX_DIR, force_rebuild: bool = False) -> bool:
    """Load FAISS index and BM25 retriever from disk/memory if exists, otherwise create them from the PDF."""
    global vector_store, bm25_retriever
    if not force_rebuild and os.path.exists(save_dir):
        try:
            vector_store = FAISS.load_local(
                save_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            # Reconstruct BM25Retriever from local FAISS docstore
            chunks = list(vector_store.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = config.RETRIEVER_K
            logger.info("Loaded FAISS index and initialized BM25 retriever from local directory '%s'", save_dir)
            return True
        except Exception as e:
            logger.error("Error loading local index: %s", e)

    if not os.path.exists(pdf_path):
        logger.error("PDF file not found at '%s'", pdf_path)
        return False

    logger.info("Loading and indexing PDF: %s ...", pdf_path)
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(save_dir)

    # Initialize BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.RETRIEVER_K
    logger.info("FAISS vector store built, BM25 retriever initialized, and saved to '%s'", save_dir)

    # Clear chat history memory when rebuilding/uploading a new PDF
    clear_rag()
    return True


def ask_rag(question: str) -> dict:
    """Execute query over FAISS/BM25 stores and LLM with chat history memory context."""
    global vector_store, bm25_retriever, memory
    if vector_store is None or bm25_retriever is None:
        return {
            "answer": "No PDF has been uploaded/indexed yet.",
            "structured_items": [],
            "sources": [],
            "token_usage": {}
        }

    # FAISS dense retriever
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    # Ensemble retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=config.ENSEMBLE_WEIGHTS
    )

    # Hybrid Search (Retrieve candidates)
    t0_hybrid = time.time()
    retrieved_docs = ensemble_retriever.invoke(question)
    hybrid_time = time.time() - t0_hybrid
    logger.info("Hybrid search retrieved %d chunks in %.3fs", len(retrieved_docs), hybrid_time)

    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant context in the PDF to answer this.",
            "structured_items": [],
            "sources": [],
            "token_usage": {}
        }

    # Reranking using Cross-Encoder
    t0_rerank = time.time()
    pairs = [(question, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(pairs)

    # Sort docs by score descending and take top N
    scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:config.RERANK_TOP_K]]
    reranked_scores = [score for doc, score in scored_docs[:config.RERANK_TOP_K]]
    rerank_time = time.time() - t0_rerank
    confidences = [1 / (1 + math.exp(-float(s))) for s in reranked_scores]
    score_conf_strs = [f"{s:.3f} ({c*100:.2f}%)" for s, c in zip(reranked_scores, confidences)]
    logger.info(
        "Reranking selected top %d chunks in %.3fs. Scores: %s. Page(s): %s",
        len(reranked_docs),
        rerank_time,
        ", ".join(score_conf_strs),
        ", ".join(str(doc.metadata.get("page", 0) + 1) for doc in reranked_docs)
    )

    context_text = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in reranked_docs
    )

    # Initialize JSON output parser with schema
    parser = JsonOutputParser(pydantic_object=LLMResponseSchema)

    # Initialize model
    llm = ChatOllama(model=config.OLLAMA_MODEL)

    # Compile prompt with memory support and JSON format instructions
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise document-analysis assistant. Answer the user's question using ONLY "
            "the information present in the context blocks below, extracted from a PDF document.\n\n"
            "Rules:\n"
            "1. Base your answer strictly on the provided context. Do not use outside knowledge.\n"
            "2. If the answer is not present in the context, set answer to: "
            "\"I don't know based on the provided document.\" and structured_items to [].\n"
            "3. If the context is only partially relevant, answer what is supported and state what is missing.\n"
            "4. Do not guess, infer beyond what is stated, or fabricate numbers, names, or facts.\n"
            "5. If context blocks conflict, mention the conflict instead of silently picking one.\n"
            "6. Keep answers concise — do not repeat the context verbatim.\n"
            "7. If the question asks for a list, table, or definitions, populate structured_items with "
            "a list of dicts (e.g. {{\"term\": \"...\", \"definition\": \"...\"}}) and set answer to ONLY "
            "a brief one-line intro (e.g. 'The glossary defines 8 terms:'). Do NOT repeat the full list in answer.\n\n"
            "{format_instructions}\n\n"
            "Context from PDF:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | llm

    # Load context history
    history_vars = memory.load_memory_variables({})
    chat_history = history_vars.get("chat_history", [])

    # Embed the retrieved chunks once for similarity checking
    chunk_texts = [doc.page_content for doc in reranked_docs]

    threshold = config.VALIDATION_THRESHOLD
    max_attempts = config.VALIDATION_MAX_ATTEMPTS


    validation_passed = False
    validation_score = 0.0
    attempts = 0

    total_token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }

    answer = ""
    structured_items = []

    while attempts < max_attempts:
        attempts += 1
        logger.info("Generation attempt %d of %d", attempts, max_attempts)

        # Prepare question: append validation feedback if it failed previously
        if attempts > 1:
            current_question = (
                f"{question}\n\n"
                f"[SYSTEM ALERT: The previous generated answer failed our validation check because its semantic "
                f"similarity score ({validation_score:.1f}) was below the required threshold of {threshold:.1f}. "
                f"Please regenerate a more precise and accurate answer that is strictly grounded in the provided "
                f"context blocks, avoiding any outside knowledge or hallucinations.]"
            )
        else:
            current_question = question

        # LLM execution
        t0_llm = time.time()
        ai_message = chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "question": current_question,
            "format_instructions": parser.get_format_instructions()
        })
        llm_time = time.time() - t0_llm
        logger.info("LLM generation completed in %.3fs for attempt %d", llm_time, attempts)

        # Accumulate token usage
        usage = ai_message.usage_metadata or {}
        total_token_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_token_usage["output_tokens"] += usage.get("output_tokens", 0)
        total_token_usage["total_tokens"] += usage.get("total_tokens", 0)

        # Parse JSON response
        try:
            parsed_response = parser.parse(ai_message.content)
        except Exception as parse_err:
            logger.warning("JSON parse failed (%s) on attempt %d, falling back to raw content.", parse_err, attempts)
            parsed_response = {"answer": ai_message.content, "structured_items": []}

        answer = parsed_response.get("answer", "")
        structured_items = parsed_response.get("structured_items", [])

        # Validate similarity
        validation_passed, validation_score = validate_answer(answer, chunk_texts, threshold)

        logger.info(
            "Validation result for attempt %d | Passed: %s | Score: %.1f | Threshold: %.1f",
            attempts, validation_passed, validation_score, threshold
        )

        if validation_passed:
            break

    # Log total token usage
    logger.info(
        "Total Token Usage | Input: %d | Output: %d | Total: %d",
        total_token_usage["input_tokens"],
        total_token_usage["output_tokens"],
        total_token_usage["total_tokens"],
    )

    # Save interaction to sliding memory
    memory.save_context({"input": question}, {"output": answer})

    # Format sources for frontend
    sources = [
        {
            "page": doc.metadata.get("page", 0) + 1,
            "content": doc.page_content,
            "score": round(float(score), 3),
            "confidence_pct": round((1 / (1 + math.exp(-float(score)))) * 100, 2)
        }
        for doc, score in scored_docs[:config.RERANK_TOP_K]
    ]

    return {
        "answer": answer,
        "structured_items": structured_items,
        "sources": sources,
        "token_usage": total_token_usage,
        "validation": {
            "passed": validation_passed,
            "score": round(validation_score, 1),
            "attempts": attempts,
            "threshold": threshold
        }
    }


def clear_rag():
    """Clear memory buffer."""
    global memory
    memory.clear()
    logger.info("Chat memory cleared")


def get_memory_contents() -> list:
    """Return memory window contents as a serializable list."""
    history_vars = memory.load_memory_variables({})
    messages = []
    for msg in history_vars.get("chat_history", []):
        messages.append({
            "type": msg.type,
            "content": msg.content
        })
    return messages
