# Hiver Support Agent

This repository contains the end-to-end customer support AI agent for the Hiver SDE Intern assignment.

## Headline Results

*(Run `make eval` to populate after labeling golden set)*

| Metric | Trivial | Simple | Agent |
|---|---|---|---|
| Intent F1 | | | |
| Esc. F1 | | | |
| Quality (Mean) | | | |

## Prerequisites

- Python 3.10+
- Kaggle API key (if downloading real dataset)
- OpenAI API Key (or Ollama installed for local model fallback)

## Setup

1. `cp .env.example .env` and fill in your keys.
2. Run `make setup` to install dependencies.

## How to Reproduce

### 1. Fast Reproduce Path (Cached)
If you just want to verify the headline numbers without hitting the LLM API:
```bash
make eval
```
*Note: Because responses are cached in `data/cache/` (if checked into source control), this will run instantly.*

### 2. Full Regenerate Path
If you want to run the pipeline from scratch (costs API credits if using OpenAI):
1. Clear the cache: `rm -rf data/cache/*`
2. Download and preprocess data: `make data`
3. Build retrieval index: `make index`
4. Label golden set: Edit `data/golden/golden_set.csv` manually.
5. Run Evaluation: `make eval`

See `REPORT.md` for full methodology and failure analysis.
