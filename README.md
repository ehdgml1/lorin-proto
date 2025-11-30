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

# Install Poetry (if not already installed)
pip install poetry

# Install dependencies
poetry install

# Download pretrained checkpoints
poetry run pip install huggingface_hub
poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('dongdonghee/lorin-aosp-checkpoints', local_dir='./checkpoints')"
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
| `LLM_MODEL` | Model name (e.g., `gemini-2.5-flash`) |

## Usage

### End-to-End Pipeline (Single File)

Process a raw logcat file through the complete pipeline (Stage 1 + Stage 2):

```bash
poetry run python main.py --log-file /path/to/logcat.log
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
poetry run python main.py
```

### Custom Query Examples

```bash
# Debug a specific issue
poetry run python main.py -q "Find logs related to memory leak in ActivityManager"

# Locate boot events
poetry run python main.py --query "When does the Android system boot process start the DropBoxManager service?"

# Combine with log processing
poetry run python main.py --log-file /path/to/logcat.log -q "Which log range shows the app crash?"
```

---

## Batch Experiment Pipeline

For running experiments on multiple log cases (e.g., benchmark datasets), LORIN provides a three-step batch processing workflow:

```
Step 1: Build FAISS Indexes    →    Step 2: Run Stage2 Analysis    →    Step 3: Evaluate Results
   (run_build_faiss.py)              (run_stage2.py)                    (evaluate_results.py)
```

### Data Directory Structure

```
data/
├── logs/                      # Input log CSV files
│   ├── case_1.csv
│   ├── case_2.csv
│   ├── ...
│   └── query.json             # Queries for each case
├── gt/                        # Ground truth files
│   ├── case_1gt.txt           # Format: line[start~end]
│   ├── case_2gt.txt
│   └── ...
└── faiss_indices/             # Generated FAISS indexes
    ├── case_1/
    │   ├── index.faiss
    │   └── index.pkl
    └── ...

results/
├── stage2_results.json        # Stage2 output
└── evaluation_results.json    # Evaluation metrics
```

---

### Step 1: Build FAISS Indexes

Build FAISS indexes for all case CSV files. The embedding model is loaded once and reused across all cases for efficiency.

```bash
poetry run python run_build_faiss.py [OPTIONS]
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `1` | GPU device ID to use |
| `--start` | `1` | First case number to process |
| `--end` | `20` | Last case number to process |
| `--mode` | `safe` | Execution mode: `safe`, `balanced`, `performance` |
| `--csv-dir` | `data/logs` | Directory containing case CSV files |
| `--output-dir` | `data/faiss_indices` | Directory for output indexes |
| `--no-skip-existing` | - | Rebuild existing indexes (default: skip) |

#### Execution Modes

| Mode | Batch Size | Description |
|------|------------|-------------|
| `safe` | 4 | Slow but memory-stable |
| `balanced` | 8 | Balance between speed and memory |
| `performance` | 16 | Fast but high memory usage |

#### Examples

```bash
# Build indexes for all 20 cases on GPU 1
poetry run python run_build_faiss.py --gpu 1 --start 1 --end 20

# Build only cases 5-10 with performance mode on GPU 0
poetry run python run_build_faiss.py --gpu 0 --start 5 --end 10 --mode performance

# Rebuild a single case (case 3)
poetry run python run_build_faiss.py --gpu 1 --start 3 --end 3 --no-skip-existing

# Custom directories
poetry run python run_build_faiss.py \
    --csv-dir /path/to/csv/files \
    --output-dir /path/to/output \
    --gpu 1
```

#### Output

- `data/faiss_indices/case_N/index.faiss` - FAISS index file
- `data/faiss_indices/case_N/index.pkl` - Metadata pickle file
- `data/faiss_indices/build_summary.json` - Build summary with timing info

---

### Step 2: Run Stage2 Analysis

Execute Stage2 (PRGG pipeline) on all cases using pre-built FAISS indexes.

```bash
poetry run python run_stage2.py [OPTIONS]
```

#### Prerequisites

1. FAISS indexes must be built first (Step 1)
2. `query.json` must exist with queries for each case
3. LLM provider must be configured in `.env`

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--start` | `1` | First case number to run |
| `--end` | `20` | Last case number to run |
| `--query-file` | `data/logs/query.json` | Path to queries JSON |
| `--faiss-base` | `data/faiss_indices` | Base directory for FAISS indexes |
| `--output` | `results/stage2_results.json` | Output file path |

#### Query File Format (`query.json`)

```json
{
  "case_1": "Analyze the system crash that occurred before the unexpected reboot",
  "case_2": "Find the memory leak in ActivityManager service",
  "case_3": "Identify the root cause of the ANR in the launcher app"
}
```

#### Examples

```bash
# Run Stage2 for all 20 cases
poetry run python run_stage2.py --start 1 --end 20

# Run only case 3
poetry run python run_stage2.py --start 3 --end 3

# Run cases 10-15 with custom output
poetry run python run_stage2.py --start 10 --end 15 --output results/batch_10_15.json

# Custom query file and FAISS directory
poetry run python run_stage2.py \
    --query-file /path/to/queries.json \
    --faiss-base /path/to/faiss_indices
```

#### Output Format (`stage2_results.json`)

```json
{
  "metadata": {
    "random_seed": 42,
    "llm_provider": "GEMINI",
    "llm_model": "gemini-2.5-flash",
    "total_time_seconds": 1234.56,
    "cases_succeeded": 19,
    "cases_failed": 1
  },
  "cases": {
    "case_1": {
      "case_id": 1,
      "query": "...",
      "recommended_ranges": [
        {"start": 1500, "end": 1650, "count": 151}
      ],
      "stage2_time_seconds": 45.2,
      "success": true
    }
  }
}
```

---

### Step 3: Evaluate Results

Calculate Coverage and Reduction metrics by comparing recommendations with ground truth.

```bash
poetry run python evaluate_results.py [OPTIONS]
```

#### Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Coverage** | `\|R̂ ∩ R*\| / \|R*\| × 100` | Percentage of ground truth covered |
| **Reduction** | `(1 - \|R̂\| / \|Original\|) × 100` | Percentage of log reduced |

Where:
- `R̂` = Recommended line ranges
- `R*` = Ground truth line ranges
- `Original` = Total lines in original log

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--results` | `results/stage2_results.json` | Stage2 results file |
| `--gt-dir` | `data/gt` | Directory containing GT files |
| `--csv-dir` | `data/logs` | Directory containing case CSVs |
| `--output` | `results/evaluation_results.json` | Output file path |

#### Ground Truth File Format (`case_Ngt.txt`)

```
line[1500~1650]
line[2000~2100]
```

#### Examples

```bash
# Evaluate with default paths
poetry run python evaluate_results.py

# Custom paths
poetry run python evaluate_results.py \
    --results results/stage2_results.json \
    --gt-dir data/ground_truth \
    --output results/eval.json
```

#### Output Format (`evaluation_results.json`)

```json
{
  "summary": {
    "total_cases": 20,
    "evaluated_cases": 20,
    "avg_coverage": 65.58,
    "avg_reduction": 91.51
  },
  "cases": {
    "case_1": {
      "case_id": 1,
      "coverage": 100.0,
      "reduction": 92.5,
      "recommended_lines": 400,
      "gt_lines": 150,
      "overlap_lines": 150,
      "original_lines": 5340
    }
  }
}
```

---

### Complete Batch Workflow Example

```bash
# Step 1: Build FAISS indexes for all cases
poetry run python run_build_faiss.py --gpu 1 --start 1 --end 20

# Step 2: Run Stage2 analysis
poetry run python run_stage2.py --start 1 --end 20

# Step 3: Evaluate results
poetry run python evaluate_results.py

# View results
cat results/evaluation_results.json | jq '.summary'
```

---

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
│   │   ├── planning/        # Query decomposition (Planner)
│   │   ├── retrieval/       # FAISS retrieval (Retriever)
│   │   ├── evaluation/      # Relevance grading (Grader)
│   │   └── generation/      # Answer synthesis (Generator)
│   ├── llm/                 # LLM backend abstraction
│   ├── make_faiss/          # Index construction
│   └── prompt/              # Jinja2 templates
├── stage1_filtering/
│   ├── models/              # MAE architecture
│   ├── scripts/             # Training & inference
│   └── stage1_process.py    # Stage 1 pipeline
├── run_build_faiss.py       # Batch FAISS index builder
├── run_stage2.py            # Batch Stage2 runner
├── evaluate_results.py      # Evaluation metrics calculator
└── main.py                  # Main entry point (Full / Stage 2)
```

## Troubleshooting

### GPU Memory Issues

If you encounter CUDA OOM errors during FAISS index building:

```bash
# Use safe mode with smaller batch size
poetry run python run_build_faiss.py --mode safe --gpu 1

# Or specify a different GPU
poetry run python run_build_faiss.py --gpu 0
```

### LLM Provider Configuration

Ensure your `.env` file has the correct settings:

```bash
# For Gemini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_api_key_here

# For OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_api_key_here
```

## License

MIT License - see [LICENSE](LICENSE) for details.
