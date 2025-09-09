# app.py
import os
import fitz  # PyMuPDF
import google.generativeai as genai
import chromadb
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# --- Environment and Configuration ---
load_dotenv()

# --- Constants ---
UPLOAD_FOLDER = "data"
CHROMA_PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# --- Initialization ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Please create a .env file and set it.")
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- PDF & RAG Logic ---
def extract_pages_text(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text: pages.append((i + 1, text))
    doc.close()
    return pages

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if chunk_size <= overlap: raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end == text_length: break
        start += chunk_size - overlap
    return chunks

def pdf_to_chunks(pdf_path: str) -> List[Dict]:
    pages = extract_pages_text(pdf_path)
    base_filename = os.path.basename(pdf_path)
    all_chunks = []
    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text)
        for i, chunk_text_content in enumerate(page_chunks):
            all_chunks.append({
                "id": f"{base_filename}_p{page_num}_c{i}",
                "text": chunk_text_content,
                "page": page_num,
                "source": base_filename
            })
    return all_chunks

class RAGPipeline:
    def __init__(self, embedding_model_name: str, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedder = SentenceTransformer(embedding_model_name)

    def _get_or_create_collection(self, collection_name: str):
        return self.client.get_or_create_collection(name=collection_name)

    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]
        
    def delete_collection(self, collection_name: str):
        self.client.delete_collection(name=collection_name)

    def ingest_pdf(self, pdf_path: str, collection_name: str) -> Dict[str, Any]:
        chunks = pdf_to_chunks(pdf_path)
        if not chunks: return {"ingested_chunks": 0, "collection": collection_name}
        docs = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [{"page": c["page"], "source": c["source"]} for c in chunks]
        embeddings = self.embedder.encode(docs, show_progress_bar=True).tolist()
        collection = self._get_or_create_collection(collection_name)
        collection.upsert(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)
        return {"ingested_chunks": len(docs), "collection": collection_name}

    def query(self, query_text: str, collection_name: str, top_k: int = 4) -> List[Dict]:
        collection = self._get_or_create_collection(collection_name)
        query_embedding = self.embedder.encode([query_text])[0].tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
        formatted_results = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            formatted_results.append({
                "text": doc,
                "metadata": results.get("metadatas", [[]])[0][i],
                "distance": results.get("distances", [[]])[0][i],
            })
        return formatted_results

    @staticmethod
    def format_sources(results: List[Dict]) -> List[str]:
        return [f"{r['metadata']['source']} - Page {r['metadata']['page']} (Similarity: {1 - r['distance']:.2f})" for r in results]

def build_prompt(question: str, contexts: List[Dict]) -> str:
    context_str = "\n\n---\n\n".join([f"Source: {c['metadata']['source']}, Page: {c['metadata']['page']}\n\n{c['text']}" for c in contexts])
    return f"""
    You are an expert Q&A assistant. Your task is to answer the user's question based *only* on the provided context documents.
    Context: {context_str}
    Question: {question}
    Instructions: Synthesize a clear and concise answer. If the context doesn't contain the answer, state that. Cite page numbers like this: [Page 5].
    Answer:
    """

def answer_question(question: str, retrieved_docs: List[Dict]) -> str:
    prompt = build_prompt(question, retrieved_docs)
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        return "An error occurred while generating the answer."

# --- Global RAG Pipeline Instance ---
rag_pipeline = RAGPipeline(embedding_model_name=EMBEDDING_MODEL_NAME, persist_directory=CHROMA_PERSIST_DIRECTORY)

# --- Flask Routes ---

@app.route("/")
def home():
    """Serves the main single-page application."""
    return render_template("index.html")

@app.route("/api/collections", methods=["GET"])
def get_collections():
    """Returns a list of existing collections."""
    collections = rag_pipeline.list_collections()
    return jsonify({"collections": collections})

@app.route("/api/upload", methods=["POST"])
def upload():
    """Handles PDF file uploads and ingestion."""
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        collection_name = request.form.get("collection", "").strip() or filename
        info = rag_pipeline.ingest_pdf(filepath, collection_name=collection_name)
        return jsonify(info)
    return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

@app.route("/api/ask", methods=["POST"])
def ask():
    """Handles user questions and returns answers from the RAG pipeline."""
    data = request.get_json()
    collection = data.get("collection", "").strip()
    question = data.get("question", "").strip()
    if not collection or not question: return jsonify({"error": "Collection and question are required"}), 400
    
    retrieved_docs = rag_pipeline.query(question, collection_name=collection)
    if not retrieved_docs:
        answer = "I couldn't find relevant information in the document to answer your question."
        sources = []
    else:
        answer = answer_question(question, retrieved_docs)
        sources = rag_pipeline.format_sources(retrieved_docs)
    return jsonify({"question": question, "answer": answer, "sources": sources})
    
@app.route("/api/delete", methods=["POST"])
def delete():
    """Deletes a specified collection."""
    data = request.get_json()
    collection_name = data.get("collection", "").strip()
    if not collection_name: return jsonify({"error": "Collection name is required"}), 400
    try:
        rag_pipeline.delete_collection(collection_name)
        return jsonify({"message": f"Collection '{collection_name}' deleted."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
