import os
import re
import json
import time
import uuid
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, stream_with_context, Response

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from transformers import pipeline
import torch

# ==============================
# LOGGING
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")

# ==============================
# CONFIG
# ==============================
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "documents"
VECTOR_STORE_DIR = BASE_DIR / "vector_stores"
CHAT_HISTORY_DIR = BASE_DIR / "chat_histories"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}
MAX_FILE_SIZE_MB = 50

for folder in [UPLOAD_FOLDER, VECTOR_STORE_DIR, CHAT_HISTORY_DIR]:
    folder.mkdir(exist_ok=True)

# ==============================
# GLOBAL STATE
# ==============================
vector_stores = {}       # doc_id -> FAISS db
document_registry = {}  # doc_id -> metadata dict
embeddings_model = None
generator = None

# ==============================
# LAZY MODEL LOADING
# ==============================
def get_embeddings():
    global embeddings_model
    if embeddings_model is None:
        logger.info("Loading embedding model...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return embeddings_model

def get_generator():
    global generator
    if generator is None:
        logger.info("Loading LLM...")
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    return generator

# ==============================
# DOCUMENT LOADING
# ==============================
def load_document(file_path: Path) -> list[Document]:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif ext == ".txt" or ext == ".md":
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def build_vector_store(doc_id: str, file_path: Path) -> dict:
    """Load, chunk, embed, and store a document. Returns metadata."""
    documents = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source_file"] = file_path.name

    emb = get_embeddings()
    db = FAISS.from_documents(chunks, emb)

    # Persist
    store_path = VECTOR_STORE_DIR / doc_id
    db.save_local(str(store_path))

    vector_stores[doc_id] = db
    return {
        "doc_id": doc_id,
        "filename": file_path.name,
        "num_chunks": len(chunks),
        "num_pages": len(documents),
        "uploaded_at": datetime.now().isoformat(),
        "size_kb": round(file_path.stat().st_size / 1024, 1)
    }

def load_existing_stores():
    """Reload persisted vector stores on startup."""
    for store_path in VECTOR_STORE_DIR.iterdir():
        if store_path.is_dir():
            doc_id = store_path.name
            try:
                emb = get_embeddings()
                db = FAISS.load_local(str(store_path), emb, allow_dangerous_deserialization=True)
                vector_stores[doc_id] = db
                logger.info(f"Reloaded store: {doc_id}")
            except Exception as e:
                logger.warning(f"Could not reload {doc_id}: {e}")

# ==============================
# RAG PIPELINE
# ==============================
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)  # remove non-printable
    return text.strip()

def rerank_docs(query: str, docs: list[Document]) -> list[Document]:
    """Simple keyword-based reranking on top of vector similarity."""
    query_terms = set(query.lower().split())
    scored = []
    for doc in docs:
        text_lower = doc.page_content.lower()
        score = sum(1 for term in query_terms if term in text_lower)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]

def build_prompt(query: str, context: str, chat_history: list[dict]) -> str:
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]  # last 4 turns
        history_text = "\n".join([
            f"User: {h['query']}\nAssistant: {h['answer']}"
            for h in recent
        ])
        history_text = f"\nPrevious conversation:\n{history_text}\n"

    query_lower = query.lower()

    if any(w in query_lower for w in ["list", "all", "every", "enumerate", "what are"]):
        style_hint = "Return a complete numbered list. Do not miss any items."
    elif any(w in query_lower for w in ["summarize", "summary", "overview", "brief"]):
        style_hint = "Provide a concise summary in 3-5 sentences."
    elif any(w in query_lower for w in ["compare", "difference", "vs", "versus"]):
        style_hint = "Compare clearly, highlighting key differences."
    elif any(w in query_lower for w in ["how", "explain", "describe"]):
        style_hint = "Explain step by step in simple, clear language."
    else:
        style_hint = "Answer directly and concisely."

    return f"""You are a helpful document assistant. Answer only using the provided context.
{history_text}
Context from document:
{context}

Instruction: {style_hint}

Question: {query}

Answer:"""

def get_answer(query: str, doc_ids: list[str], chat_history: list[dict]) -> dict:
    """Multi-document RAG answer with source attribution."""
    start = time.time()

    if not doc_ids:
        return {"answer": "No documents selected. Please upload and select a document.", "sources": [], "latency_ms": 0}

    # Retrieve from all selected stores
    all_docs = []
    for doc_id in doc_ids:
        db = vector_stores.get(doc_id)
        if db:
            docs = db.similarity_search(query, k=6)
            all_docs.extend(docs)

    if not all_docs:
        return {"answer": "No relevant content found in the selected documents.", "sources": [], "latency_ms": 0}

    # Rerank
    all_docs = rerank_docs(query, all_docs)

    # Extract special patterns first
    combined_raw = " ".join(d.page_content for d in all_docs[:6])
    if "email" in query.lower():
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", combined_raw)
        if emails:
            return {
                "answer": f"Email address found: **{emails[0]}**",
                "sources": _extract_sources(all_docs[:2]),
                "latency_ms": int((time.time() - start) * 1000)
            }

    if any(w in query.lower() for w in ["phone", "mobile", "contact number"]):
        phones = re.findall(r"\+?\d[\d\s\-]{8,}\d", combined_raw)
        if phones:
            return {
                "answer": f"Phone number found: **{phones[0].strip()}**",
                "sources": _extract_sources(all_docs[:2]),
                "latency_ms": int((time.time() - start) * 1000)
            }

    # Build context from top chunks
    context = "\n\n---\n\n".join([clean_text(d.page_content) for d in all_docs[:5]])
    prompt = build_prompt(query, context, chat_history)

    gen = get_generator()
    result = gen(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]

    # Clean up output
    answer = result.split("Answer:")[-1].strip()
    lines = list(dict.fromkeys(answer.split("\n")))
    answer = "\n".join(lines).strip()

    if not answer or len(answer) < 5:
        answer = "I couldn't find a clear answer in the document. Try rephrasing your question."

    return {
        "answer": answer,
        "sources": _extract_sources(all_docs[:3]),
        "latency_ms": int((time.time() - start) * 1000)
    }

def _extract_sources(docs: list[Document]) -> list[dict]:
    seen = set()
    sources = []
    for doc in docs:
        key = f"{doc.metadata.get('source_file', 'unknown')}_{doc.metadata.get('page', '?')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:120] + "..."
            })
    return sources

# ==============================
# CHAT HISTORY HELPERS
# ==============================
def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def load_chat_history(session_id: str) -> list[dict]:
    path = CHAT_HISTORY_DIR / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []

def save_chat_history(session_id: str, history: list[dict]):
    path = CHAT_HISTORY_DIR / f"{session_id}.json"
    path.write_text(json.dumps(history[-50:], indent=2))  # keep last 50 turns

# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Check size
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE_MB}MB"}), 400

    # Generate deterministic doc_id from filename + content hash
    content = file.read()
    doc_id = hashlib.md5(content).hexdigest()[:12]
    file.seek(0)

    # Save file
    safe_name = re.sub(r"[^\w\-_\.]", "_", file.filename)
    file_path = UPLOAD_FOLDER / f"{doc_id}_{safe_name}"
    file.save(str(file_path))

    try:
        meta = build_vector_store(doc_id, file_path)
        document_registry[doc_id] = meta
        return jsonify({"success": True, "document": meta})
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500

@app.route("/api/documents", methods=["GET"])
def list_documents():
    docs = list(document_registry.values())
    return jsonify({"documents": docs})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "").strip()
    doc_ids = data.get("doc_ids", [])

    if not query:
        return jsonify({"error": "Empty query"}), 400

    session_id = get_session_id()
    history = load_chat_history(session_id)

    result = get_answer(query, doc_ids, history)

    # Save to history
    history.append({
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "timestamp": datetime.now().isoformat(),
        "doc_ids": doc_ids
    })
    save_chat_history(session_id, history)

    return jsonify({
        "answer": result["answer"],
        "sources": result["sources"],
        "latency_ms": result["latency_ms"],
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/history", methods=["GET"])
def get_history():
    session_id = get_session_id()
    history = load_chat_history(session_id)
    return jsonify({"history": history})

@app.route("/api/history", methods=["DELETE"])
def clear_history():
    session_id = get_session_id()
    path = CHAT_HISTORY_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
    return jsonify({"success": True})

@app.route("/api/document/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    if doc_id in document_registry:
        del document_registry[doc_id]
    if doc_id in vector_stores:
        del vector_stores[doc_id]
    store_path = VECTOR_STORE_DIR / doc_id
    if store_path.exists():
        import shutil
        shutil.rmtree(store_path)
    return jsonify({"success": True})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "documents_loaded": len(vector_stores),
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    })

# ==============================
# STARTUP
# ==============================
if __name__ == "__main__":
    logger.info("Initializing models...")
    get_embeddings()
    get_generator()
    load_existing_stores()
    logger.info(f"Loaded {len(vector_stores)} existing vector stores")
    app.run(debug=False, host="0.0.0.0", port=5001)
