# LORIN

**Log Retrieval with Intelligent Decomposition and Narrowing**

This repository contains the implementation of LORIN, a two-stage hybrid pipeline for Log Range Recommendation (LRR) in Android system debugging. Given a bug report and its associated logcat dump, LORIN identifies the most relevant log segments that help developers diagnose the issue.

## Overview

LORIN addresses the challenge of analyzing large-scale Android logs (often 10K+ lines) by combining anomaly detection with retrieval-augmented generation:

- **Stage 1**: Teacher-Student Masked Autoencoder filters candidate log regions based on reconstruction error patterns
- **Stage 2**: PRGG (Planner-Retriever-Grader-Generator) pipeline performs semantic retrieval with query decomposition and temporal narrowing

> **Note**: Stage 1 is modular and can be replaced with any anomaly detection model that outputs per-line anomaly scores. The PRGG pipeline (Stage 2) operates independently on the filtered candidate set.

## Architecture

```
Bug Report + Logcat
        │
        ▼
┌───────────────────┐
│  Stage 1: MAE     │  Anomaly-based filtering
│  Teacher-Student  │  (Reverse Knowledge Distillation)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stage 2: PRGG    │
│  ├─ Planner       │  Query decomposition
│  ├─ Retriever     │  FAISS + BGE-multilingual-gemma2
│  ├─ Grader        │  Relevance evaluation
│  └─ Generator     │  Final recommendation
└───────────────────┘
        │
        ▼
  Recommended Log Range
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- ~8GB disk space for model checkpoints

## Installation

```bash
git clone https://github.com/ehdgml1/lorin-proto.git
cd lorin-proto

# Install dependencies
pip install poetry
poetry install

# Download pretrained checkpoints
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('dongdonghee/lorin-aosp-checkpoints', local_dir='./checkpoints')"
```

## Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description |
|----------|-------------|
| `FAISS_INDEX_PATH` | Path to FAISS index directory |
| `LLM_PROVIDER` | LLM backend (`gemini`, `openai`, `exaone`) |
| `GEMINI_API_KEY` | API key if using Gemini |

## Usage

### End-to-End Pipeline (Full)

Process a raw logcat file through the complete pipeline (Stage 1 + Stage 2):

```bash
python main.py --log-file /path/to/logcat.log
```

Optional arguments:

| Argument | Description |
|----------|-------------|
| `-q`, `--query` | User query for log analysis (uses default if not specified) |
| `--stage1-output-dir` | Parsing output directory (default: `<log_dir>/result`) |
| `--stage1-anomaly-dir` | Anomaly detection output (default: `<output-dir>/anomaly_detection`) |
| `--faiss-dir` | FAISS index directory (default: `<anomaly-dir>/faiss_index`) |

### Stage 2 Only (PRGG Pipeline)

If you already have a FAISS index from a previous run:

```bash
export FAISS_INDEX_PATH=/path/to/faiss_index
python main.py
```

### Custom Query Examples

```bash
# Debug a specific issue
python main.py -q "Find logs related to memory leak in ActivityManager"

# Locate boot events
python main.py --query "When does the Android system boot process start the DropBoxManager service?"

# Combine with log processing
python main.py --log-file /path/to/logcat.log -q "Which log range shows the app crash?"
```

## Input Format

LORIN expects standard Android logcat output:

```
01-15 12:34:56.789  1234  5678 D ActivityManager: Start proc ...
01-15 12:34:56.790  1234  5678 I WindowManager: Screen frozen ...
```

The Drain parser extracts structured fields (timestamp, PID, TID, level, component, message) automatically.

## Output

The system outputs:

1. **Labeled CSV**: Log lines with anomaly scores from Stage 1
2. **Recommended Range**: `line[start~end]` format indicating relevant log segments
3. **Explanation**: Natural language reasoning for the recommendation

## Model Checkpoints

### Stage 1: Anomaly Detection

Pretrained weights are available on Hugging Face:

**Repository**: [dongdonghee/lorin-aosp-checkpoints](https://huggingface.co/dongdonghee/lorin-aosp-checkpoints)

| File | Description | Size |
|------|-------------|------|
| `teacher_pretrained_aosp.pt` | Teacher MAE (6-layer Transformer) | 216MB |
| `revkd_out_v5_aosp/seed*/best.pt` | Student networks (3 seeds) | 61MB each |

### Stage 2: Embedding Model

The PRGG pipeline uses [BGE-multilingual-gemma2](https://huggingface.co/BAAI/bge-multilingual-gemma2) for semantic retrieval (~3GB). It will be downloaded automatically on first run.

To download manually, clone from the Hugging Face repository and set the path in `.env`:

```bash
# Download model
git lfs install
git clone https://huggingface.co/BAAI/bge-multilingual-gemma2

# Set local path in .env
BGE_MODEL_NAME=./bge-multilingual-gemma2
```

## Project Structure

```
lorin-proto/
├── LORIN/
│   ├── agent/
│   │   ├── planning/      # Query decomposition (Planner)
│   │   ├── retrieval/     # FAISS retrieval (Retriever)
│   │   ├── evaluation/    # Relevance grading (Grader)
│   │   └── generation/    # Answer synthesis (Generator)
│   ├── llm/               # LLM backend abstraction
│   ├── make_faiss/        # Index construction
│   └── prompt/            # Jinja2 templates
├── stage1_filtering/
│   ├── models/            # MAE architecture
│   ├── scripts/           # Training & inference
│   └── stage1_process.py  # Stage 1 pipeline
└── main.py                # Main entry point (Full / Stage 2)
```

## License

MIT License - see [LICENSE](LICENSE) for details.
