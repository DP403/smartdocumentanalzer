import os
import json
import time
import flask
from flask_cors import CORS
from google import genai
import pypdf
import numpy as np
import faiss
import hashlib
from werkzeug.utils import secure_filename

app = flask.Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SECURITY NOTE: Move your API Key to an environment variable (.env) for production safety
GOOGLE_API_KEY = "AIzaSyCHe2JIyg45XTYExLFvrr04y9O3y_IuoAY"  
client = genai.Client(api_key=GOOGLE_API_KEY)

GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

# The vector store starts empty now until a user uploads a file
vector_store = {}

def get_pdf_hash(path):
    if not os.path.exists(path):
        return None
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Core processing pipeline converted into a dynamic helper function
def process_pdf_file(target_file_path):
    global vector_store
    
    current_hash = get_pdf_hash(target_file_path)
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, f"vector_store_{current_hash}.index")
    CHUNKS_JSON_PATH = os.path.join(BASE_DIR, f"chunks_{current_hash}.json")

    # 1. Check if this specific file configuration is cached
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_JSON_PATH):
        print("\n⚡ Found matching local cache! Loading directly...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            text_chunks = json.load(f)
        vector_store = {"index": index, "chunks": text_chunks}
        return "Cache matched! Loaded instantly with zero API overhead."

    # 2. Clear out older, stale vector cache configurations
    print("\n🔄 New PDF configuration detected. Evicting old vector cache...")
    for filename in os.listdir(BASE_DIR):
        if (filename.startswith("vector_store") and filename.endswith(".index")) or \
           (filename.startswith("chunks") and filename.endswith(".json")):
            if filename != os.path.basename(FAISS_INDEX_PATH) and filename != os.path.basename(CHUNKS_JSON_PATH):
                try:
                    os.remove(os.path.join(BASE_DIR, filename))
                except Exception as e:
                    print(f"⚠️ Cache clean up skipped for {filename}: {e}")

    # 3. Parse text from the new upload file
    full_text = ""
    reader = pypdf.PdfReader(target_file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    # 4. Slicing/Chunking text
    text_chunks = []
    for i in range(0, len(full_text), 900):
        text_chunks.append(full_text[i:i+1000])

    if not text_chunks:
        raise ValueError("The uploaded document contains no clear text content to parse.")

    # 5. Build vector embedding indexes via API call blocks
    chunk_embeddings = []
    batch_size = 20
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=batch
        )
        batch_vectors = [embedding.values for embedding in result.embeddings]
        chunk_embeddings.extend(batch_vectors)
        if i + batch_size < len(text_chunks):
            time.sleep(2)

    # 6. Build and commit data to FAISS engine memory
    dimension = len(chunk_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings).astype('float32'))
    vector_store = {"index": index, "chunks": text_chunks}

    # Save things locally to disk for tracking
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(text_chunks, f, ensure_ascii=False)
        
    return f"Processing completed successfully. Created {len(text_chunks)} text fragments."

# --- API Endpoints ---

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'file' not in flask.request.files:
        return flask.jsonify({"error": "Missing file field in request payload"}), 400
        
    uploaded_file = flask.request.files['file']
    if uploaded_file.filename == '':
        return flask.jsonify({"error": "No selected file details found"}), 400

    if uploaded_file and uploaded_file.filename.lower().endswith('.pdf'):
        try:
            filename = secure_filename(uploaded_file.filename)
            saved_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(saved_path)
            
            # Execute processing logic dynamically on file receipt
            message = process_pdf_file(saved_path)
            
            # Clean up the raw PDF file from disk right after processing to save storage space
            if os.path.exists(saved_path):
                os.remove(saved_path)
                
            return flask.jsonify({"status": "Success", "message": message})
        except Exception as e:
            return flask.jsonify({"error": f"Failed to process file upload: {str(e)}"}), 500
            
    return flask.jsonify({"error": "Invalid file format. Only PDF files are supported."}), 400

@app.route('/api/query', methods=['POST'])
def handle_query():
    # Defensive check: block query if a vector store hasn't been built yet
    if not vector_store or "index" not in vector_store:
        return flask.jsonify({"error": "No context document loaded. Please upload a PDF through the interface first."}), 400

    data = flask.request.json or {}
    user_query = data.get('query', '')
    
    if not user_query:
        return flask.jsonify({"error": "Empty Query Submitted"}), 400
        
    try:
        relevant_chunks = find_relevant_chunks(user_query)
        answer = generate_answer(user_query, relevant_chunks)
        return flask.jsonify({"answer": answer})
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500

def find_relevant_chunks(query, k=3):
    query_embedding_result = client.models.embed_content(
        model=EMBEDDING_MODEL_NAME,
        contents=query
    )
    query_embedding = query_embedding_result.embeddings[0].values
    query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    distances, indices = vector_store["index"].search(query_embedding_np, k)
    return [vector_store["chunks"][i] for i in indices[0]]

def generate_answer(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = client.models.generate_content(
        model=GENERATIVE_MODEL_NAME,
        contents=prompt
    )
    return response.text if response and response.text else "Error gathering response content."

if __name__ == '__main__':
    # Notice: we removed the static initialization step from here!
    app.run(port=5000, debug=True, use_reloader=False)