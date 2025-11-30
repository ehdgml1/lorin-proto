#!/usr/bin/env python3
"""
LORIN FAISS Index Builder
=========================
Builds separate FAISS indexes for each case CSV file (case_1.csv ~ case_20.csv).
Each case gets its own index directory (case_1, case_2, ...).

Usage:
    poetry run python run_build_faiss.py --gpu 1 --start 1 --end 20
"""

# ============================================================
# 중요: GPU 설정은 반드시 torch import 전에 해야 함!
# ============================================================
import os
import sys

# GPU 설정 (기본값: GPU 1)
DEFAULT_GPU_ID = 1


def setup_environment(gpu_id: int):
    """torch import 전에 환경변수 설정"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(f"🎯 Using GPU {gpu_id} only (CUDA_VISIBLE_DEVICES={gpu_id})")


# argparse로 GPU ID를 먼저 파싱
import argparse
parser = argparse.ArgumentParser(description="Build FAISS indexes for case CSV files")
parser.add_argument("--gpu", type=int, default=DEFAULT_GPU_ID, help="GPU device ID to use (default: 1)")
parser.add_argument("--start", type=int, default=1, help="Start case number (default: 1)")
parser.add_argument("--end", type=int, default=20, help="End case number (default: 20)")
parser.add_argument("--mode", choices=["safe", "balanced", "performance"], default="safe", help="Execution mode")
parser.add_argument("--no-skip-existing", action="store_true", help="Rebuild existing indexes")
parser.add_argument("--csv-dir", type=str, default="/home/bigdata/1113/lorin-proto/data/logs", help="CSV directory")
parser.add_argument("--output-dir", type=str, default="/home/bigdata/1113/lorin-proto/data/faiss_indices", help="Output directory")

args = parser.parse_args()

# 환경변수 설정 (torch import 전!)
setup_environment(args.gpu)

# 이제 torch와 다른 모듈 import
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import pandas as pd

# LORIN 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))


def check_gpu_memory() -> bool:
    """GPU 메모리 상태 확인"""
    if not torch.cuda.is_available():
        print("⚠️  GPU not available - will use CPU mode")
        return False

    print("\n=== GPU Memory Status ===")
    device_count = torch.cuda.device_count()

    if device_count == 0:
        print("⚠️  No GPU visible - check CUDA_VISIBLE_DEVICES setting")
        return False

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        free = total - allocated

        print(f"GPU Device {i}: {props.name}")
        print(f"  Total: {total:.1f} GB")
        print(f"  Used: {allocated:.1f} GB")
        print(f"  Free: {free:.1f} GB")

        if free < 8:
            print(f"  ⚠️  Warning: Low free memory!")

    return True


def build_single_index(
    csv_path: Path,
    index_path: Path,
    embeddings,
    skip_existing: bool = True
) -> Tuple[bool, str, float]:
    """
    단일 CSV 파일에 대한 FAISS 인덱스 빌드

    Args:
        csv_path: CSV 파일 경로
        index_path: 인덱스 저장 경로
        embeddings: 재사용할 임베딩 모델
        skip_existing: 이미 존재하면 건너뛰기

    Returns:
        Tuple of (success, message, duration_seconds)
    """
    from LORIN.make_faiss.make_faiss_with_bge_multilingual_gemma2 import build_docs_from_df
    from langchain_community.vectorstores import FAISS

    start_time = datetime.now()

    # 이미 존재하면 건너뛰기
    if skip_existing and index_path.exists():
        faiss_file = index_path / "index.faiss"
        if faiss_file.exists():
            return True, "Already exists (skipped)", 0.0

    # CSV 로드
    try:
        df = pd.read_csv(csv_path)
        row_count = len(df)
        print(f"  Loaded {row_count:,} rows from CSV")
    except Exception as e:
        return False, f"CSV load failed: {e}", 0.0

    # FAISS 인덱스 빌드
    try:
        print(f"  Building documents from CSV...")
        docs = build_docs_from_df(df)
        print(f"  Generated {len(docs):,} document chunks")

        print(f"  Creating FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 저장
        index_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving index to {index_path}...")
        vectorstore.save_local(str(index_path))

        # 검증
        faiss_file = index_path / "index.faiss"
        pkl_file = index_path / "index.pkl"

        duration = (datetime.now() - start_time).total_seconds()

        if faiss_file.exists() and pkl_file.exists():
            faiss_size = faiss_file.stat().st_size / 1024**2
            return True, f"Success ({row_count:,} rows, {len(docs):,} chunks, {faiss_size:.1f} MB)", duration
        else:
            return False, "Index files not found after build", duration

    except torch.cuda.OutOfMemoryError as e:
        return False, f"OOM error: {e}", (datetime.now() - start_time).total_seconds()

    except Exception as e:
        return False, f"Build error: {e}", (datetime.now() - start_time).total_seconds()

    finally:
        # 로컬 변수만 삭제 (embeddings는 유지!)
        if 'vectorstore' in locals():
            del vectorstore
        if 'docs' in locals():
            del docs
        if 'df' in locals():
            del df

        # GPU 캐시만 비우기
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def build_all_case_indexes(
    csv_dir: Path,
    output_dir: Path,
    start_case: int,
    end_case: int,
    mode: str = "safe",
    skip_existing: bool = True
) -> Dict:
    """
    모든 케이스에 대한 FAISS 인덱스 빌드

    핵심: 임베딩 모델을 한 번만 로드하여 모든 케이스에서 재사용
    """
    # 모드 설정
    configs = {
        "safe": {"batch_size": 4, "description": "Safe mode (slow but stable)"},
        "balanced": {"batch_size": 8, "description": "Balanced mode"},
        "performance": {"batch_size": 16, "description": "Performance mode (fast but high memory)"}
    }

    config = configs.get(mode, configs["safe"])
    print(f"\n🔧 Mode: {config['description']}")
    print(f"   Batch size: {config['batch_size']}")

    # ============================================================
    # 핵심: 임베딩 모델을 한 번만 로드
    # ============================================================
    print(f"\n🚀 Loading embeddings model (once for all cases)...")
    from LORIN.make_faiss.make_faiss_with_bge_multilingual_gemma2 import (
        BGEGemma2MultiGPUEmbeddings,
        MODEL_NAME
    )

    try:
        embeddings = BGEGemma2MultiGPUEmbeddings(
            model_name=MODEL_NAME,
            batch_size_per_gpu=config["batch_size"],
            use_fp16=True
        )
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {}

    # 결과 요약
    summary = {
        "start_time": datetime.now().isoformat(),
        "mode": mode,
        "skip_existing": skip_existing,
        "gpu_id": args.gpu,
        "results": []
    }

    # 케이스별 빌드
    total_cases = end_case - start_case + 1
    for case_num in range(start_case, end_case + 1):
        idx = case_num - start_case + 1
        print(f"\n{'='*60}")
        print(f"[{idx}/{total_cases}] Processing: case_{case_num}")
        print(f"{'='*60}")

        csv_path = csv_dir / f"case_{case_num}.csv"
        index_path = output_dir / f"case_{case_num}"

        print(f"CSV: {csv_path}")
        print(f"Index: {index_path}")

        # CSV 존재 확인
        if not csv_path.exists():
            print(f"⚠️  CSV not found: {csv_path}")
            summary["results"].append({
                "case_id": case_num,
                "csv_path": str(csv_path),
                "index_path": str(index_path),
                "success": False,
                "message": "CSV not found",
                "duration_seconds": 0
            })
            continue

        # 빌드 실행
        success, message, duration = build_single_index(
            csv_path=csv_path,
            index_path=index_path,
            embeddings=embeddings,
            skip_existing=skip_existing
        )

        # 결과 기록
        result = {
            "case_id": case_num,
            "csv_path": str(csv_path),
            "index_path": str(index_path),
            "success": success,
            "message": message,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat()
        }
        summary["results"].append(result)

        # 결과 출력
        status = "✅" if success else "❌"
        print(f"\n{status} Result: {message}")
        if duration > 0:
            print(f"Duration: {duration:.1f}s")

    # ============================================================
    # 모든 케이스 완료 후 임베딩 모델 정리
    # ============================================================
    print("\n🧹 Cleaning up embeddings model...")
    del embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("✅ Cleanup complete")

    # 요약 정보
    summary["end_time"] = datetime.now().isoformat()
    summary["total_count"] = len(summary["results"])
    summary["success_count"] = sum(1 for r in summary["results"] if r["success"])
    summary["failed_count"] = summary["total_count"] - summary["success_count"]

    return summary


def print_summary(summary: Dict):
    """빌드 요약 출력"""
    print(f"\n{'='*60}")
    print("BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {summary['total_count']}")
    print(f"Successful: {summary['success_count']} ✅")
    print(f"Failed: {summary['failed_count']} ❌")
    print(f"Mode: {summary['mode']}")
    print(f"GPU: {summary['gpu_id']}")

    if summary['failed_count'] > 0:
        print("\nFailed builds:")
        for result in summary['results']:
            if not result['success']:
                print(f"  ❌ case_{result['case_id']}: {result['message']}")

    print("\nSuccessful builds:")
    total_time = 0
    for result in summary['results']:
        if result['success']:
            print(f"  ✅ case_{result['case_id']}: {result['message']} ({result['duration_seconds']}s)")
            total_time += result['duration_seconds']

    print(f"\nTotal build time: {total_time:.1f}s")


def main():
    """메인 실행"""
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)

    if not csv_dir.exists():
        print(f"❌ CSV directory not found: {csv_dir}")
        sys.exit(1)

    print("🚀 LORIN FAISS Index Builder")
    print("="*60)
    print(f"CSV directory: {csv_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Cases: {args.start} ~ {args.end}")

    # GPU 상태 확인
    check_gpu_memory()

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 빌드 실행
    summary = build_all_case_indexes(
        csv_dir=csv_dir,
        output_dir=output_dir,
        start_case=args.start,
        end_case=args.end,
        mode=args.mode,
        skip_existing=not args.no_skip_existing
    )

    if not summary:
        print("❌ Build failed")
        sys.exit(1)

    # 요약 저장
    summary_path = output_dir / "build_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Summary saved to: {summary_path}")

    # 요약 출력
    print_summary(summary)

    # 종료 코드
    if summary['failed_count'] > 0:
        print(f"\n⚠️  {summary['failed_count']} build(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All {summary['success_count']} builds completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
