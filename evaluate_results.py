#!/usr/bin/env python3
"""
LORIN Evaluation Script
=======================
Calculates Coverage and Reduction metrics by comparing
system-recommended line ranges with ground truth (GT).

Coverage = |R_hat ∩ R*| / |R*| × 100
Reduction = (1 - |R_hat| / |OriginalLog|) × 100

Usage:
    poetry run python evaluate_results.py --results results/stage2_results.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple


def parse_gt_file(gt_path: Path) -> List[Tuple[int, int]]:
    """
    GT 파일에서 라인 범위 파싱

    형식: line[start~end] 또는 line[start-end]
    """
    ranges = []

    if not gt_path.exists():
        return ranges

    with open(gt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # line[a~b] 또는 line[a-b] 패턴 찾기
    pattern = r'line\s*\[\s*(\d+)\s*[~\-]\s*(\d+)\s*\]'
    matches = re.findall(pattern, content, re.IGNORECASE)

    for match in matches:
        start, end = int(match[0]), int(match[1])
        if start <= end:
            ranges.append((start, end))

    return ranges


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    겹치는 범위 병합

    예: [(1, 100), (50, 150)] -> [(1, 150)]
    """
    if not ranges:
        return []

    # 시작점 기준 정렬
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        last = merged[-1]

        # 겹치거나 연속되면 병합
        if current[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def ranges_to_set(ranges: List[Tuple[int, int]]) -> Set[int]:
    """범위 리스트를 라인 번호 집합으로 변환"""
    line_set = set()
    for start, end in ranges:
        line_set.update(range(start, end + 1))
    return line_set


def count_lines_in_ranges(ranges: List[Tuple[int, int]]) -> int:
    """범위의 총 라인 수 계산 (겹치는 부분 제외)"""
    merged = merge_ranges(ranges)
    total = 0
    for start, end in merged:
        total += end - start + 1
    return total


def calculate_metrics(
    recommended_ranges: List[Dict],
    gt_ranges: List[Tuple[int, int]],
    original_log_lines: int
) -> Dict:
    """
    Coverage와 Reduction 계산

    Args:
        recommended_ranges: 시스템 추천 범위 [{"start": a, "end": b}, ...]
        gt_ranges: GT 범위 [(start, end), ...]
        original_log_lines: 원본 로그 라인 수

    Returns:
        {"coverage": float, "reduction": float, ...}
    """
    # 추천 범위를 튜플 리스트로 변환
    rec_tuples = [(r["start"], r["end"]) for r in recommended_ranges]

    # 겹치는 부분 병합
    rec_merged = merge_ranges(rec_tuples)
    gt_merged = merge_ranges(gt_ranges)

    # 라인 집합으로 변환
    rec_set = ranges_to_set(rec_merged)
    gt_set = ranges_to_set(gt_merged)

    # 교집합 (겹치는 라인)
    overlap_set = rec_set & gt_set

    # 각 집합 크기
    rec_count = len(rec_set)
    gt_count = len(gt_set)
    overlap_count = len(overlap_set)

    # Coverage = |R_hat ∩ R*| / |R*| × 100
    coverage = (overlap_count / gt_count * 100) if gt_count > 0 else 0.0

    # Reduction = (1 - |R_hat| / |OriginalLog|) × 100
    reduction = (1 - rec_count / original_log_lines) * 100 if original_log_lines > 0 else 0.0

    return {
        "coverage": round(coverage, 2),
        "reduction": round(reduction, 2),
        "recommended_lines": rec_count,
        "gt_lines": gt_count,
        "overlap_lines": overlap_count,
        "original_lines": original_log_lines,
        "recommended_ranges_merged": rec_merged,
        "gt_ranges_merged": gt_merged
    }


def get_csv_line_count(csv_dir: Path, case_id: int) -> int:
    """CSV 파일의 라인 수 (헤더 제외)"""
    csv_path = csv_dir / f"case_{case_id}.csv"
    if not csv_path.exists():
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        # 헤더 제외
        return sum(1 for _ in f) - 1


def evaluate_all_cases(
    results_path: Path,
    gt_dir: Path,
    csv_dir: Path
) -> Dict:
    """모든 케이스 평가"""

    # Stage2 결과 로드
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    evaluation = {
        "summary": {
            "total_cases": 0,
            "evaluated_cases": 0,
            "avg_coverage": 0.0,
            "avg_reduction": 0.0
        },
        "cases": {}
    }

    coverage_sum = 0.0
    reduction_sum = 0.0
    evaluated_count = 0

    for case_key, case_data in results.get("cases", {}).items():
        case_id = case_data.get("case_id")
        if case_id is None:
            continue

        evaluation["summary"]["total_cases"] += 1

        # GT 파일 로드
        gt_path = gt_dir / f"case_{case_id}gt.txt"
        gt_ranges = parse_gt_file(gt_path)

        if not gt_ranges:
            evaluation["cases"][case_key] = {
                "case_id": case_id,
                "error": "GT file not found or empty",
                "coverage": None,
                "reduction": None
            }
            continue

        # 추천 범위
        recommended_ranges = case_data.get("recommended_ranges", [])

        if not recommended_ranges:
            evaluation["cases"][case_key] = {
                "case_id": case_id,
                "error": "No recommended ranges",
                "coverage": 0.0,
                "reduction": 100.0,
                "gt_lines": count_lines_in_ranges(gt_ranges)
            }
            continue

        # 원본 로그 라인 수
        original_lines = get_csv_line_count(csv_dir, case_id)

        # 메트릭 계산
        metrics = calculate_metrics(recommended_ranges, gt_ranges, original_lines)

        evaluation["cases"][case_key] = {
            "case_id": case_id,
            "coverage": metrics["coverage"],
            "reduction": metrics["reduction"],
            "recommended_lines": metrics["recommended_lines"],
            "gt_lines": metrics["gt_lines"],
            "overlap_lines": metrics["overlap_lines"],
            "original_lines": metrics["original_lines"],
            "success": case_data.get("success", False)
        }

        coverage_sum += metrics["coverage"]
        reduction_sum += metrics["reduction"]
        evaluated_count += 1

    # 평균 계산
    if evaluated_count > 0:
        evaluation["summary"]["evaluated_cases"] = evaluated_count
        evaluation["summary"]["avg_coverage"] = round(coverage_sum / evaluated_count, 2)
        evaluation["summary"]["avg_reduction"] = round(reduction_sum / evaluated_count, 2)

    return evaluation


def print_evaluation_table(evaluation: Dict):
    """평가 결과 테이블 출력"""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Case':<10} {'Coverage(%)':<12} {'Reduction(%)':<12} {'Overlap':<10} {'GT Lines':<10} {'Rec Lines':<10} {'Original':<10}")
    print("-" * 100)

    for case_key in sorted(evaluation["cases"].keys(), key=lambda x: int(x.split('_')[1])):
        case = evaluation["cases"][case_key]
        case_id = f"case_{case['case_id']}"

        if case.get("error"):
            print(f"{case_id:<10} {'N/A':<12} {'N/A':<12} {'-':<10} {'-':<10} {'-':<10} {case.get('error', '')}")
        else:
            coverage = f"{case['coverage']:.1f}"
            reduction = f"{case['reduction']:.1f}"
            overlap = str(case.get('overlap_lines', '-'))
            gt_lines = str(case.get('gt_lines', '-'))
            rec_lines = str(case.get('recommended_lines', '-'))
            original = str(case.get('original_lines', '-'))

            print(f"{case_id:<10} {coverage:<12} {reduction:<12} {overlap:<10} {gt_lines:<10} {rec_lines:<10} {original:<10}")

    print("-" * 100)
    summary = evaluation["summary"]
    print(f"{'Average':<10} {summary['avg_coverage']:<12.1f} {summary['avg_reduction']:<12.1f}")
    print("=" * 100)
    print(f"\nEvaluated: {summary['evaluated_cases']} / {summary['total_cases']} cases")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LORIN Stage2 results")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/home/bigdata/1113/lorin-proto/results/stage2_results.json"),
        help="Path to stage2_results.json"
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/home/bigdata/1113/lorin-proto/data/gt"),
        help="Directory containing GT files"
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("/home/bigdata/1113/lorin-proto/data/logs"),
        help="Directory containing case CSV files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/bigdata/1113/lorin-proto/results/evaluation_results.json"),
        help="Output JSON file path"
    )

    args = parser.parse_args()

    if not args.results.exists():
        print(f"❌ Results file not found: {args.results}")
        print("   Run stage2 first: poetry run python run_stage2.py")
        return

    print("🔍 LORIN Evaluation")
    print("=" * 60)
    print(f"Results: {args.results}")
    print(f"GT dir: {args.gt_dir}")
    print(f"CSV dir: {args.csv_dir}")

    # 평가 실행
    evaluation = evaluate_all_cases(
        results_path=args.results,
        gt_dir=args.gt_dir,
        csv_dir=args.csv_dir
    )

    # 결과 저장
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Evaluation saved to: {args.output}")

    # 테이블 출력
    print_evaluation_table(evaluation)


if __name__ == "__main__":
    main()
