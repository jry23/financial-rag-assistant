# Financial RAG — MVP

Goal: answer questions about Apple's most recent 10-K using retrieval-augmented generation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

Set a user agent for SEC EDGAR:
```bash
export SEC_USER_AGENT="Your Name your@email.com"
```

## Run order (first time)

```bash
python -m ingestion.download_10k          # pulls Apple 10-K to data/raw/
python -m preprocessing.clean_and_chunk   # HTML -> sections -> chunks in data/processed/
python -m embeddings.build_index          # embeds chunks, saves FAISS index
python -m app.cli "What were Apple's main risk factors?"
```

## Run eval

```bash
python -m evaluation.run_eval             # runs all questions in evaluation/test_set.json
```

## Project layout

Each module is small and runnable on its own so you can debug one layer at a time.

## Notes on what's deliberately NOT here (yet)

- No hybrid retrieval (BM25). Add in v2 only if eval shows keyword queries failing.
- No reranking. Same — add if eval shows top-k ordering is bad.
- No query rewriting. Same.
- No structured data (GARCH/PCA) fusion. That's the v2 differentiator.

The point of v1 is to have a working baseline + an eval set, so we can measure whether v2 changes actually help.
