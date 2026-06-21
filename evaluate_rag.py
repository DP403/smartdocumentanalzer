import os  # evaluate_rag.py
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.run_config import RunConfig
from google import genai
from google.genai import errors as genai_errors
from app import find_relevant_chunks, generate_answer, load_index_from_disk

# 1. Initialize client and factories
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
ragas_llm = llm_factory(model="gemini-2.5-flash", client=client, provider="google")
_google_embeddings = GoogleEmbeddings(model="gemini-embedding-001", client=client)

try:
    from ragas.embeddings.base import BaseRagasEmbeddings as _RagasEmbeddingsBase
except ImportError:
    _RagasEmbeddingsBase = object


class LegacyEmbeddingsAdapter(_RagasEmbeddingsBase):
    """
    Classic ragas metrics (e.g. AnswerRelevancy imported from ragas.metrics)
    expect a BaseRagasEmbeddings subclass with embed_query()/embed_documents().
    The modern GoogleEmbeddings class only exposes embed_text()/embed_texts(),
    so this adapter bridges the two. Subclassing BaseRagasEmbeddings (instead of
    just duck-typing) ensures internal isinstance(metric, MetricWithEmbeddings)
    / isinstance(embeddings, BaseRagasEmbeddings) checks inside ragas pass,
    so the adapter doesn't get silently swapped out for a default embedder.
    """

    def __init__(self, modern_embeddings):
        try:
            super().__init__()
        except TypeError:
            pass
        self._embeddings = modern_embeddings

    def embed_query(self, text):
        return self._embeddings.embed_text(text)

    def embed_documents(self, texts):
        return self._embeddings.embed_texts(texts)

    async def aembed_query(self, text):
        return await self._embeddings.aembed_text(text)

    async def aembed_documents(self, texts):
        return await self._embeddings.aembed_texts(texts)


ragas_embeddings = LegacyEmbeddingsAdapter(_google_embeddings)

# 2. Seed evaluation set — replace these with questions relevant to your PDF
test_data = {
    "question": [
        "What is the main topic of the document?",
        "What conclusions does the document reach?",
    ],
    "answer": [],
    "contexts": [],
}


def call_with_retry(fn, *args, max_attempts=5, base_delay=10, **kwargs):
    """Retry a Gemini API call on transient 503/UNAVAILABLE errors with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            print(f"⚠️  Server overloaded ({e}). Retry {attempt}/{max_attempts} in {wait}s...")
            time.sleep(wait)


def run_evaluation():
    if not load_index_from_disk():
        print("❌ Error: No index found on disk.")
        return

    print("🚀 Starting RAG evaluation...")

    for query in test_data["question"]:
        context = call_with_retry(find_relevant_chunks, query)        # list[str] of chunks
        answer = call_with_retry(generate_answer, query, context)
        print(f"DEBUG: Processed query: {query} | Context length: {len(context)}")
        test_data["answer"].append(answer)
        test_data["contexts"].append(context)           # don't double-wrap in a list
        time.sleep(2)  # small buffer between rows to avoid per-minute bursts

    dataset = Dataset.from_dict(test_data)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
    ]

    # max_workers=1 forces RAGAS to make API calls one at a time instead of in
    # parallel bursts, which is gentler on a tight quota (fewer simultaneous
    # in-flight requests even though it doesn't change the total request count).
    run_config = RunConfig(max_workers=1)

    print("DEBUG: Loop finished. Computing scores now...")
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
    )
    print("\n--- Evaluation Results ---")
    print(scores)


if __name__ == "__main__":
    run_evaluation()