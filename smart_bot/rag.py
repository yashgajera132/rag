import os
import sys
from pathlib import Path

# Add root directory to sys.path so config can be imported regardless of execution context
try:
    import config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import CrossEncoder

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
            print(f"[RAG] Loaded FAISS index and initialized BM25 retriever from local directory '{save_dir}'")
            return True
        except Exception as e:
            print(f"[RAG] Error loading local index: {e}")

    if not os.path.exists(pdf_path):
        print(f"[RAG] Error: PDF file not found at '{pdf_path}'")
        return False

    print(f"[RAG] Loading and indexing PDF: {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(save_dir)
    
    # Initialize BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.RETRIEVER_K
    print(f"[RAG] FAISS vector store built, BM25 retriever initialized, and saved to '{save_dir}'")
    
    # Clear chat history memory when rebuilding/uploading a new PDF
    clear_rag()
    return True


def ask_rag(question: str) -> dict:
    """Execute query over FAISS/BM25 stores and LLM with chat history memory context."""
    global vector_store, bm25_retriever, memory
    if vector_store is None or bm25_retriever is None:
        return {
            "answer": "No PDF has been uploaded/indexed yet.",
            "sources": []
        }

    # FAISS dense retriever
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    # Ensemble retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=config.ENSEMBLE_WEIGHTS
    )

    # Hybrid Search (Retrieve candidates)
    retrieved_docs = ensemble_retriever.invoke(question)
    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant context in the PDF to answer this.",
            "sources": []
        }

    # Reranking using Cross-Encoder
    pairs = [(question, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    
    # Sort docs by score descending and take top 3
    scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:config.RERANK_TOP_K]]

    context_text = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in reranked_docs
    )

    # Initialize model
    llm = ChatOllama(
        model=config.OLLAMA_MODEL
    )

    # Compile prompt with memory support
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise document-analysis assistant. Answer the user's question using ONLY "
        "the information present in the context blocks below, extracted from a PDF document.\n\n"
        "Rules:\n"
        "1. Base your answer strictly on the provided context. Do not use outside knowledge.\n"
        "2. If the answer is not present in the context, respond exactly with: "
        "\"I don't know based on the provided document.\"\n"
        "3. If the context is only partially relevant, answer what is supported and state what is missing.\n"
        "4. Do not guess, infer beyond what is stated, or fabricate numbers, names, or facts.\n"
        "5. If context blocks conflict, mention the conflict instead of silently picking one.\n"
        "6. Keep answers concise — do not repeat the context verbatim.\n"
        "7. If the question asks for a list, table, or specific format, follow that format exactly.\n\n"
        "Context from PDF:\n{context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    # Load context history
    history_vars = memory.load_memory_variables({})
    chat_history = history_vars.get("chat_history", [])

    # LLM execution (1 API Call)
    answer = chain.invoke({
        "context": context_text,
        "chat_history": chat_history,
        "question": question
    })

    # Save interaction to sliding memory
    memory.save_context({"input": question}, {"output": answer})

    # Format sources for frontend
    sources = [
        {
            "page": doc.metadata.get("page", 0) + 1,
            "content": doc.page_content
        }
        for doc in reranked_docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }



def clear_rag():
    """Clear memory buffer."""
    global memory
    memory.clear()
    print("[RAG] Chat memory cleared")


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
