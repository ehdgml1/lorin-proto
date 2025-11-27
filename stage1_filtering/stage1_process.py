#!/usr/bin/env python3
"""
Parse a raw Android log file and immediately run the Stage 1 anomaly
detection pipeline (scripts/test.py) with one command.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

from Drain import LogParser
from scripts.test import (
    build_dual_branch_and_save,
    infer_ensemble_unlabeled_and_label_rows,
)

# Drain parser defaults (aligned with parse_log.py)
ANDROID_LOG_FORMAT = r"<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>"
ANDROID_REGEX = [
    r"(/[\w-]+)+",
    r"([\w-]+\.){2,}[\w-]+",
    r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
]
LOG_NAME = "Android"
DEFAULT_THRESHOLD = 5
DEFAULT_DELIMITER: list[str] = []

# Checkpoint defaults (paths relative to this file)
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER_CKPT = ROOT_DIR / "teacher_pretrained_aosp.pt"
DEFAULT_STUDENT_ROOT = ROOT_DIR / "revkd_out_v5_aosp"
LABELED_CSV_NAME = "Android.log_structured_clean.labeled.csv"
DEFAULT_FAISS_DIR_NAME = "faiss_index"


@dataclass(frozen=True)
class PipelineResult:
    structured_csv: Path
    templates_csv: Path
    labeled_csv: Path
    parse_dir: Path
    anomaly_dir: Path
    faiss_index_dir: Path | None = None


def _resolve_dir(candidate: Path | None, fallback: Path) -> Path:
    """Return an absolute, expanded directory path."""
    return (candidate.expanduser() if candidate else fallback).resolve()


def _ensure_file(path: Path, reason: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{reason}: {path}")


def _validate_models(teacher_ckpt: Path, student_root: Path | None, ckpt_files: str) -> None:
    _ensure_file(teacher_ckpt, "Teacher checkpoint is missing")
    if ckpt_files.strip():
        return  # explicit ckpt file list supplied; student_root not required
    if student_root is not None:
        _ensure_file(student_root, "Student checkpoint root is missing")


def parse_with_drain(
    log_path: Path,
    output_dir: Path,
    threshold: int = DEFAULT_THRESHOLD,
    log_format: str = ANDROID_LOG_FORMAT,
    regex: list[str] = ANDROID_REGEX,
    delimiter: list[str] | None = None,
) -> Tuple[Path, Path]:
    """Run the Drain parser and return generated CSV paths."""
    log_path = log_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _ensure_file(log_path, "Log file not found")
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = LogParser(
        logname=LOG_NAME,
        log_format=log_format,
        indir=str(log_path.parent),
        outdir=str(output_dir),
        threshold=threshold,
        delimeter=DEFAULT_DELIMITER if delimiter is None else delimiter,
        rex=regex,
    )
    parser.parse(log_path.name)

    structured_csv = output_dir / f"{log_path.name}_structured.csv"
    templates_csv = output_dir / f"{log_path.name}_templates.csv"

    _ensure_file(structured_csv, "Structured CSV was not created")
    _ensure_file(templates_csv, "Templates CSV was not created")

    return structured_csv, templates_csv


def _build_pipeline_args(
    struct_csv: Path,
    tmpl_csv: Path,
    anomaly_out: Path,
    args: argparse.Namespace,
    teacher_ckpt: Path,
    student_root: Path | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        struct_csv=struct_csv,
        template_csv=tmpl_csv,
        out=anomaly_out,
        embed_model=args.embed_model,
        sbert_batch=args.sbert_batch,
        window_size=args.window_size,
        stride=args.stride,
        teacher_ckpt=teacher_ckpt,
        batch=args.batch,
        student_root=student_root,
        student_ckpt_files=args.student_ckpt_files,
        seeds=args.seeds,
        ckpts=args.ckpts,
    )


def build_faiss_from_labeled_csv(
    labeled_csv: Path,
    output_dir: Path,
    batch_size_per_gpu: int = 4,
) -> Path:
    """Build a FAISS index from the labeled CSV."""
    import gc
    import pandas as pd
    import torch
    from langchain_community.vectorstores import FAISS
    from LORIN.make_faiss.make_faiss_with_bge_multilingual_gemma2 import (
        BGEGemma2MultiGPUEmbeddings,
        build_docs_from_df,
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labeled_csv, dtype={"EventId": str})
    docs = build_docs_from_df(df)

    embeddings = BGEGemma2MultiGPUEmbeddings(
        model_name="BAAI/bge-multilingual-gemma2",
        use_multi_gpu=False,
        batch_size_per_gpu=batch_size_per_gpu,
        use_fp16=True,
    )

    vectorstore = None
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(str(output_dir))
    finally:
        if vectorstore is not None:
            del vectorstore
        del docs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    return output_dir


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """Parse the log and run anomaly detection."""
    log_file: Path = args.log_file.expanduser().resolve()
    _ensure_file(log_file, "Log file not found")

    parse_out = _resolve_dir(args.output_dir, log_file.parent / "result")
    anomaly_out = _resolve_dir(args.anomaly_out, parse_out / "anomaly_detection")

    print(f"[1/3] Parsing log file with Drain: {log_file}")
    struct_csv, tmpl_csv = parse_with_drain(
        log_file,
        parse_out,
        threshold=args.threshold,
    )
    print(f"[DONE] Parsed CSVs → {struct_csv.name}, {tmpl_csv.name}")

    teacher_ckpt = args.teacher_ckpt.expanduser().resolve()
    student_root = args.student_root.expanduser().resolve() if args.student_root else None
    _validate_models(teacher_ckpt, student_root, args.student_ckpt_files)

    anomaly_out.mkdir(parents=True, exist_ok=True)
    pipeline_args = _build_pipeline_args(
        struct_csv, tmpl_csv, anomaly_out, args, teacher_ckpt, student_root
    )

    print(f"[2/3] Building windows & embeddings → {anomaly_out}")
    N_dedup, orig_to_dedup_idx, win_starts = build_dual_branch_and_save(pipeline_args)

    print("[3/3] Running ensemble inference + labeling")
    infer_ensemble_unlabeled_and_label_rows(
        pipeline_args, N_dedup, orig_to_dedup_idx, win_starts
    )

    labeled_csv = anomaly_out / LABELED_CSV_NAME
    _ensure_file(labeled_csv, "Labeled CSV not found; check the pipeline logs")

    faiss_dir = None
    if args.build_faiss:
        faiss_dir = _resolve_dir(
            args.faiss_dir,
            anomaly_out / DEFAULT_FAISS_DIR_NAME,
        )
        print(f"[FAISS] Building index at {faiss_dir}")
        faiss_dir = build_faiss_from_labeled_csv(
            labeled_csv,
            faiss_dir,
            batch_size_per_gpu=args.faiss_batch,
        )
        os.environ["FAISS_INDEX_PATH"] = str(faiss_dir)
        print(f"[FAISS] FAISS_INDEX_PATH set to {faiss_dir}")

    if args.run_main:
        try:
            from main import main as lorin_main
        except ImportError as exc:
            raise ImportError("Unable to import top-level main.py for chaining") from exc
        print("[MAIN] Launching top-level LORIN main()")
        asyncio.run(lorin_main())

    return PipelineResult(
        structured_csv=struct_csv,
        templates_csv=tmpl_csv,
        labeled_csv=labeled_csv,
        parse_dir=parse_out,
        anomaly_dir=anomaly_out,
        faiss_index_dir=faiss_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a .log file then run stage1_filtering/scripts/test.py end-to-end."
    )
    parser.add_argument("log_file", type=Path, help="Path to the .log file to parse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store parsed templates/structured CSVs (default: <log_dir>/result)",
    )
    parser.add_argument(
        "--anomaly-out",
        type=Path,
        default=None,
        help="Directory to store anomaly detection outputs (default: <output-dir>/anomaly_detection)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Drain template split threshold",
    )

    # Inference parameters (mirrors scripts/test.py defaults)
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SBERT model name",
    )
    parser.add_argument("--sbert-batch", type=int, default=64, help="SBERT batch size")
    parser.add_argument("--window-size", type=int, default=50, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=50, help="Sliding window stride")
    parser.add_argument("--batch", type=int, default=192, help="Inference batch size")

    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=DEFAULT_TEACHER_CKPT,
        help="Teacher checkpoint path",
    )
    parser.add_argument(
        "--student-root",
        type=Path,
        default=DEFAULT_STUDENT_ROOT,
        help="Student checkpoint root containing seed{sid}/best.pt ...",
    )
    parser.add_argument(
        "--student-ckpt-files",
        type=str,
        default="",
        help="Comma-separated student ckpt paths (overrides --student-root/--seeds/--ckpts)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,77,99",
        help="Comma-separated student seed list",
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        default="best",
        help="Student ckpt tags to load (e.g., best,last,swa) or auto",
    )
    parser.add_argument(
        "--build-faiss",
        action="store_true",
        help="Build a FAISS index from the labeled CSV",
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=None,
        help="Directory to save the FAISS index (default: <anomaly-out>/faiss_index)",
    )
    parser.add_argument(
        "--faiss-batch",
        type=int,
        default=4,
        help="Batch size per GPU when building FAISS embeddings",
    )
    parser.add_argument(
        "--run-main",
        action="store_true",
        help="Run the top-level main.py flow after setting FAISS_INDEX_PATH",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline(args)
    print(f"[DONE] Labeled CSV saved to: {result.labeled_csv}")


if __name__ == "__main__":
    main()
