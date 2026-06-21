# Smart Document Analyzer

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using natural language. Upload a PDF, ask questions about it, and get grounded answers backed by semantic search over the document's content.

Built as part of an AI Developer internship project, with a custom evaluation harness using [RAGAS](https://github.com/explodinggradients/ragas) to quantitatively measure answer quality.

## How it works

1. **Upload** — a PDF is parsed page-by-page with `pypdf` and split into overlapping ~1000-character text chunks.
2. **Embed** — each chunk is embedded with Google's `gemini-embedding-001` model and stored in a [FAISS](https://github.com/facebookresearch/faiss) `IndexFlatL2` vector index.
3. **Cache** — the index and chunks are hashed and cached to disk per-PDF, so re-uploading the same file skips re-embedding entirely.
4. **Query** — a user's question is embedded the same way, the top-k most similar chunks are retrieved from FAISS, and those chunks are passed as context to `gemini-2.5-flash` to generate a grounded answer.

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index
                                                              │
User Query → Embedding → FAISS Similarity Search → Top-K Chunks
                                                              │
                                                              ▼
                                          Context + Query → Gemini 2.5 Flash → Answer
```

## Tech stack

| Layer | Tool |
|---|---|
| Backend | Python 3.9, Flask, Flask-CORS |
| LLM | Google Gemini API (`gemini-2.5-flash`) |
| Embeddings | Google Gemini API (`gemini-embedding-001`) |
| Vector store | FAISS (`IndexFlatL2`) |
| PDF parsing | pypdf |
| Evaluation | RAGAS (Faithfulness, Answer Relevancy) |

## Setup

### Prerequisites

- Python 3.9+ (a virtual environment is strongly recommended)
- A [Gemini API key](https://ai.google.dev/) — free tier works, with quota limits noted below

### Installation

```bash
# Clone the repo
git clone https://github.com/DP403/smartdocumentanalzer.git
cd smartdocumentanalzer

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

The app loads this automatically via `python-dotenv`. Never commit `.env` to git — it's already covered by `.gitignore`.

### Running the app

```bash
python app.py
```

The Flask server starts on `http://localhost:5000`.

**Endpoints:**

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a PDF (`multipart/form-data`, field name `file`). Parses, chunks, embeds, and indexes it. |
| `POST` | `/api/query` | Ask a question (`{"query": "..."}` JSON body). Returns a grounded answer from the currently loaded document. |

Example query via curl:

```bash
curl -X POST http://localhost:5000/api/upload -F "file=@your_document.pdf"
curl -X POST http://localhost:5000/api/query -H "Content-Type: application/json" -d "{\"query\": \"What is this document about?\"}"
```

## Evaluation

The RAG pipeline's answer quality is measured with [RAGAS](https://github.com/explodinggradients/ragas), using two metrics:

- **Faithfulness** — does the generated answer stay grounded in the retrieved context, without hallucinating facts not present in the source?
- **Answer Relevancy** — does the generated answer actually address the question asked?

### Latest results

| Metric | Score |
|---|---|
| Faithfulness | **1.0000** |
| Answer Relevancy | **0.8324** |

A perfect Faithfulness score indicates the model's answers are fully grounded in retrieved document content with no hallucination on the evaluated question set. Answer Relevancy of 0.83 indicates strong (not perfect) alignment between generated answers and the questions asked — typical for free-form generation rather than extractive QA.

### Running the evaluation yourself

```bash
python evaluate_rag.py
```

This loads the most recently cached FAISS index, runs a fixed set of test questions through the live retrieval + generation pipeline, then scores the results with RAGAS. Edit the `test_data["question"]` list in `evaluate_rag.py` to evaluate against your own question set.

**Note on API quota:** the free tier of the Gemini API enforces both a per-minute (5 requests/min) and a per-day (20 requests/day) limit on `gemini-2.5-flash`. The evaluation script includes automatic retry with exponential backoff for transient `503`/`429` errors, but a fully exhausted *daily* quota will require waiting for the reset (~midnight Pacific time) rather than retrying within the same run.

## Project structure

```
smartdocumentanalzer/
├── app.py              # Flask app: upload, query, RAG pipeline
├── evaluate_rag.py      # RAGAS evaluation harness
├── requirements.txt
├── .env                 # API key (not committed)
├── uploads/              # Temp storage for uploaded PDFs (cleared after processing)
└── vector_store_*.index  # Cached FAISS indices, one per unique PDF (by content hash)
└── chunks_*.json         # Cached text chunks, paired with each index
```

## Known limitations / future work

- Single-document context only — no multi-document or cross-document querying yet.
- Free-tier API quota constrains how large or how frequently documents can be evaluated/queried.
- Fixed chunk size (1000 chars, 100-char overlap) rather than semantic/structure-aware chunking.
- No authentication or multi-user isolation — single shared `vector_store` in memory.

## License

MIT