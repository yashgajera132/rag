import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import CrossEncoder

load_dotenv()

# Embedding model shared for index building and retrieval
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": False}
)

vector_store = None

# Initialize Cross-Encoder reranker globally
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Sliding window memory keeping the last 3 QA turns
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True
)


def load_or_build_index(pdf_path: str, save_dir: str = "faiss_index_local", force_rebuild: bool = False) -> bool:
    """Load FAISS index from disk if exists, otherwise create it from the PDF."""
    global vector_store
    if not force_rebuild and os.path.exists(save_dir):
        try:
            vector_store = FAISS.load_local(
                save_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[RAG] Loaded FAISS index from local directory '{save_dir}'")
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
    print(f"[RAG] FAISS vector store built and saved to '{save_dir}'")
    
    # Clear chat history memory when rebuilding/uploading a new PDF
    clear_rag()
    return True


def ask_rag(question: str) -> dict:
    """Execute query over FAISS store and LLM with chat history memory context."""
    global vector_store, memory
    if vector_store is None:
        return {
            "answer": "No PDF has been uploaded/indexed yet.",
            "sources": []
        }

    # Similarity search (Retrieve k=8 candidates)
    retrieved_docs = vector_store.similarity_search(question, k=8)
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
    reranked_docs = [doc for doc, score in scored_docs[:3]]

    context_text = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in reranked_docs
    )

    # Initialize model
    llm = ChatGroq(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    # Compile prompt with memory support
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Use ONLY the provided context blocks to answer the question. "
            "If the answer cannot be found in the context blocks, say you don't know.\n\n"
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
