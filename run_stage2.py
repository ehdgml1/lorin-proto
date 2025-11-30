#!/usr/bin/env python3
"""
LORIN Stage2 Runner
===================
Runs Stage2 (LORIN LLM analysis) for each case using pre-built FAISS indexes.

Prerequisites:
    1. FAISS indexes must be built first using run_build_faiss.py
    2. query.json must contain queries for each case

Usage:
    poetry run python run_stage2.py --start 1 --end 20
"""

import argparse
import asyncio
import json
import os
import re
import time
import gc
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 재현성을 위한 Seed 설정
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """재현성을 위한 모든 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


# Import LORIN modules
from LORIN.llm.chatbot import Chatbot, LLMProvider
from LORIN.logger.logger import Logger, get_logger
from LORIN.process.base import main_process
from LORIN.agent.state import get_last_message

# Logger 설정
try:
    logger_configurator = Logger(name="LORIN_STAGE2")
except Warning:
    pass


def parse_line_ranges(text: str) -> List[Dict[str, int]]:
    """
    LLM 응답에서 라인 범위 파싱

    지원 형식:
    - line[a~b], line[a-b]
    - lines [a~b], lines [a-b]
    - [a~b] standalone
    """
    ranges = []
    patterns = [
        r'line\s*\[\s*(\d+)\s*[~\-]\s*(\d+)\s*\]',
        r'lines?\s*\[\s*(\d+)\s*[~\-]\s*(\d+)\s*\]',
        r'lines?\s*(\d+)\s*[~\-]\s*(\d+)',
        r'\[\s*(\d+)\s*[~\-]\s*(\d+)\s*\]',
    ]

    seen = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            start, end = int(match[0]), int(match[1])
            if (start, end) not in seen and start <= end:
                seen.add((start, end))
                ranges.append({
                    "start": start,
                    "end": end,
                    "count": end - start + 1
                })

    ranges.sort(key=lambda x: x["start"])
    return ranges


def build_vectorstore(faiss_index_path: str) -> Tuple[str, Path]:
    """FAISS 인덱스 경로 확인"""
    logger = get_logger(__name__)
    index_path = Path(faiss_index_path)
    faiss_index_file = index_path / "index.faiss"

    if not faiss_index_file.exists():
        logger.warning(f"FAISS index not found at {faiss_index_file}")
    else:
        logger.info(f"FAISS index ready at {faiss_index_file}")

    return str(index_path), index_path


async def run_stage2_for_case(
    case_id: int,
    query: str,
    faiss_index_path: str,
    chatbot: Chatbot
) -> Dict[str, Any]:
    """단일 케이스에 대해 Stage2 실행"""
    logger = get_logger(__name__)

    result = {
        "case_id": case_id,
        "query": query,
        "faiss_index_path": faiss_index_path,
        "recommended_ranges": [],
        "raw_response": "",
        "stage2_time_seconds": 0.0,
        "success": False,
        "error": None
    }

    start_time = time.time()

    try:
        # FAISS 인덱스 경로 설정
        os.environ["FAISS_INDEX_PATH"] = faiss_index_path
        vectorstore, _ = build_vectorstore(faiss_index_path)

        # Stage2 실행
        states = await main_process(
            chatbot,
            vectorstore,
            question=query
        )

        # 응답 추출
        if states and len(states) > 0:
            last_state = states[-1]
            last_message = get_last_message(last_state)

            if last_message:
                response_text = last_message.content
                result["raw_response"] = response_text

                # 라인 범위 파싱
                ranges = parse_line_ranges(response_text)
                result["recommended_ranges"] = ranges

                logger.info(f"Parsed {len(ranges)} line ranges from response")
                for r in ranges:
                    logger.info(f"  - line[{r['start']}~{r['end']}] ({r['count']} lines)")

        result["success"] = True

    except Exception as e:
        logger.error(f"Stage2 failed: {e}")
        result["error"] = str(e)

    result["stage2_time_seconds"] = round(time.time() - start_time, 2)
    return result


async def run_single_case(
    case_id: int,
    query: str,
    faiss_index_path: str,
    chatbot: Chatbot
) -> Dict[str, Any]:
    """단일 케이스 실행"""
    logger = get_logger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Case {case_id}")
    logger.info(f"{'='*60}")
    logger.info(f"Query: {query[:100]}...")

    result = {
        "case_id": case_id,
        "query": query,
        "faiss_index_path": faiss_index_path,
        "stage2_time_seconds": 0.0,
        "recommended_ranges": [],
        "raw_response": "",
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None
    }

    # FAISS 인덱스 존재 확인
    faiss_file = Path(faiss_index_path) / "index.faiss"
    if not faiss_file.exists():
        result["error"] = f"FAISS index not found: {faiss_index_path}"
        logger.error(result["error"])
        return result

    # Stage2 실행
    stage2_result = await run_stage2_for_case(
        case_id=case_id,
        query=query,
        faiss_index_path=faiss_index_path,
        chatbot=chatbot
    )

    # 결과 병합
    result["stage2_time_seconds"] = stage2_result["stage2_time_seconds"]
    result["recommended_ranges"] = stage2_result["recommended_ranges"]
    result["raw_response"] = stage2_result["raw_response"]
    result["success"] = stage2_result["success"]

    if stage2_result["error"]:
        result["error"] = stage2_result["error"]

    logger.info(f"Case {case_id} completed:")
    logger.info(f"  Time: {result['stage2_time_seconds']}s")
    logger.info(f"  Ranges found: {len(result['recommended_ranges'])}")

    return result


async def run_all_cases(
    query_file: Path,
    faiss_base_dir: Path,
    output_path: Path,
    start_case: int = 1,
    end_case: int = 20
) -> Dict[str, Any]:
    """모든 케이스 실행"""
    logger = get_logger(__name__)

    # 쿼리 로드
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Chatbot 초기화
    provider_str = os.getenv("LLM_PROVIDER", "gemini").upper()
    model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    try:
        provider = LLMProvider[provider_str]
    except KeyError:
        logger.error(f"Invalid LLM_PROVIDER: {provider_str}")
        raise ValueError(f"Unsupported LLM provider: {provider_str}")

    chatbot = Chatbot(
        provider=provider,
        model=model_name,
        temperature=0.4
    )

    logger.info(f"Using LLM: {provider_str} / {model_name}")

    # 결과 컨테이너
    all_results = {
        "metadata": {
            "random_seed": RANDOM_SEED,
            "llm_provider": provider_str,
            "llm_model": model_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_time_seconds": 0.0,
            "cases_run": 0,
            "cases_succeeded": 0,
            "cases_failed": 0
        },
        "cases": {}
    }

    total_start = time.time()

    # 케이스별 실행
    for case_num in range(start_case, end_case + 1):
        case_key = f"case_{case_num}"

        # 쿼리 확인
        if case_key not in queries:
            logger.warning(f"Query not found for {case_key}, skipping...")
            continue

        query = queries[case_key]

        # FAISS 인덱스 경로
        faiss_index_path = faiss_base_dir / f"case_{case_num}"

        # 케이스 실행
        result = await run_single_case(
            case_id=case_num,
            query=query,
            faiss_index_path=str(faiss_index_path),
            chatbot=chatbot
        )

        all_results["cases"][case_key] = result
        all_results["metadata"]["cases_run"] += 1

        if result["success"]:
            all_results["metadata"]["cases_succeeded"] += 1
        else:
            all_results["metadata"]["cases_failed"] += 1

        # 중간 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Intermediate results saved to {output_path}")

    total_end = time.time()
    all_results["metadata"]["end_time"] = datetime.now().isoformat()
    all_results["metadata"]["total_time_seconds"] = round(total_end - total_start, 2)

    # 최종 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("STAGE2 COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total cases run: {all_results['metadata']['cases_run']}")
    logger.info(f"Succeeded: {all_results['metadata']['cases_succeeded']}")
    logger.info(f"Failed: {all_results['metadata']['cases_failed']}")
    logger.info(f"Total time: {all_results['metadata']['total_time_seconds']}s")
    logger.info(f"Results saved to: {output_path}")

    return all_results


def print_summary(results: Dict[str, Any]):
    """결과 요약 출력"""
    print("\n" + "="*100)
    print("STAGE2 RESULTS SUMMARY")
    print("="*100)
    print(f"{'Case':<10} {'Status':<8} {'Time(s)':<10} {'Ranges':<8} {'Line Ranges'}")
    print("-"*100)

    for case_key in sorted(results["cases"].keys(), key=lambda x: int(x.split('_')[1])):
        case = results["cases"][case_key]
        status = "OK" if case["success"] else "FAIL"
        stage2_time = case["stage2_time_seconds"]
        num_ranges = len(case["recommended_ranges"])

        ranges_str = ""
        if case["recommended_ranges"]:
            ranges_str = ", ".join([
                f"[{r['start']}~{r['end']}]"
                for r in case["recommended_ranges"][:3]
            ])
            if num_ranges > 3:
                ranges_str += f" +{num_ranges-3} more"

        print(f"{case_key:<10} {status:<8} {stage2_time:<10.2f} {num_ranges:<8} {ranges_str}")

    print("-"*100)
    meta = results["metadata"]
    print(f"Total: {meta['cases_run']} cases | "
          f"Success: {meta['cases_succeeded']} | "
          f"Failed: {meta['cases_failed']} | "
          f"Time: {meta['total_time_seconds']}s")
    print("="*100)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Run LORIN Stage2 on multiple cases"
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=Path("/home/bigdata/1113/lorin-proto/data/logs/query.json"),
        help="Path to query.json file"
    )
    parser.add_argument(
        "--faiss-base",
        type=str,
        default="/home/bigdata/1113/lorin-proto/data/faiss_indices",
        help="Base directory containing FAISS indices"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/bigdata/1113/lorin-proto/results/stage2_results.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First case number to run (default: 1)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=20,
        help="Last case number to run (default: 20)"
    )

    args = parser.parse_args()

    print("🚀 LORIN Stage2 Runner")
    print("="*60)
    print(f"Query file: {args.query_file}")
    print(f"FAISS base: {args.faiss_base}")
    print(f"Output: {args.output}")
    print(f"Cases: {args.start} ~ {args.end}")

    # Seed 설정
    set_seed(RANDOM_SEED)

    # 출력 디렉토리 생성
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 실행
    results = asyncio.run(run_all_cases(
        query_file=args.query_file,
        faiss_base_dir=Path(args.faiss_base),
        output_path=args.output,
        start_case=args.start,
        end_case=args.end
    ))

    # 요약 출력
    print_summary(results)


if __name__ == "__main__":
    main()
