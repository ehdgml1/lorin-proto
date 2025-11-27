"""
Routing Decision Module - Quality-Based Routing Logic
======================================================

LORIN 시스템의 품질 평가 결과 기반 라우팅 결정 로직입니다.
평가 결과를 분석하여 replanner, retrieve, answer 중 적절한 경로를 결정합니다.

주요 기능:
- 서브쿼리 성공/실패 분석
- 반복 라우팅 결정 로직
- 루프 방지 시스템 통합
- 재시도 전략 결정
"""

from typing import Dict, List, Any, Tuple

from ...logger.logger import get_logger
from ...config.settings import get_settings
from .loop_prevention import (
    initialize_loop_prevention,
    detect_routing_loop,
    detect_oscillation_pattern,
    log_forced_termination,
    update_loop_prevention_metadata,
)
from ..state import AgentState
from ..schema import (
    FailedQueryInfo,
    LoopPreventionData,
    RoutingDecision,
    IterationHistoryEntry,
    QueryEvaluationInfo,
    MetadataManager,
)

logger = get_logger(__name__)

# ── 중앙 집중식 설정 로드 ───────────────────────────────────
_settings = get_settings()


# ─────────────────────────────────────────────────────────
# Query Result Analysis
# ─────────────────────────────────────────────────────────


def analyze_query_results(
    cumulative_results: Dict[str, QueryEvaluationInfo]
) -> Tuple[List[FailedQueryInfo], List[str], int]:
    """평가 결과를 분석하여 실패/성공 쿼리 분류

    Args:
        cumulative_results: 누적 평가 결과

    Returns:
        Tuple[List[FailedQueryInfo], List[str], int]: (실패 쿼리들, 성공 쿼리 ID들, 전체 쿼리 수)
    """
    failed_queries = []
    successful_queries = []
    total_queries = 0

    for query_id, result_data in cumulative_results.items():
        total_queries += 1
        is_relevant = result_data.get("is_relevant", False)
        relevance_score = result_data.get("relevance_score", 0.0)
        confidence = result_data.get("confidence", 0.0)

        if not is_relevant:
            failed_queries.append({
                "query_id": query_id,
                "query_text": result_data.get("query_text", ""),
                "relevance_score": relevance_score,
                "confidence": confidence,
                "reasoning": result_data.get("reasoning", "")
            })
        else:
            successful_queries.append(query_id)

    return failed_queries, successful_queries, total_queries


def identify_persistent_failures(
    current_failures: List[FailedQueryInfo],
    iteration_history: List[IterationHistoryEntry]
) -> List[str]:
    """연속적으로 실패하는 쿼리들 식별

    Args:
        current_failures: 현재 실패한 쿼리들
        iteration_history: 반복 히스토리

    Returns:
        List[str]: 영속적으로 실패하는 쿼리 ID들
    """
    if len(iteration_history) < 2:
        return []

    current_failed_ids = {q.get("query_id") for q in current_failures}

    # 최근 반복에서의 실패 쿼리들
    recent_iteration = iteration_history[-1]
    previous_failed_count = recent_iteration.get("failed_count", 0)

    # 간단한 휴리스틱: 실패 수가 감소하지 않고 있다면 영속적 실패로 간주
    if previous_failed_count > 0 and len(current_failed_ids) >= previous_failed_count:
        return list(current_failed_ids)

    return []


# ─────────────────────────────────────────────────────────
# Routing Decision Logic
# ─────────────────────────────────────────────────────────


def should_retry_routing(state: AgentState) -> str:
    """평가 결과를 기반으로 재시도 여부 결정 (무한 루프 방지 시스템)

    이 함수는 quality_evaluator_node 실행 후 호출되어,
    관련성이 낮다고 평가된 서브쿼리들에 대해 재시도가 필요한지 판단합니다.

    **무한 루프 방지 기능:**
    - 글로벌 라우팅 반복 횟수 추적 (최대 25회)
    - 라우팅 패턴 탐지 (oscillation 감지)
    - 강제 종료 조건 및 우아한 성능 저하
    - 실시간 루프 감지 및 즉시 중단

    라우팅 결정:
    - "replanner": 실패한 서브쿼리 재구성 필요
    - "retrieve": 직접 재검색 (재구성 완료된 쿼리)
    - "answer": 모든 평가 통과 또는 루프 감지 시 강제 종료

    Args:
        state: quality_evaluator 결과가 포함된 agent state

    Returns:
        str: 'replanner', 'retrieve', 또는 'answer'
    """
    logger.info("[should_retry_routing] Starting routing decision with loop prevention")

    try:
        # MetadataManager를 통한 안전한 메타데이터 접근
        evaluator_data = MetadataManager.safe_get_metadata(state, "quality_evaluator") or {}
        replanner_data = MetadataManager.safe_get_metadata(state, "replanner") or {}

        # 안전장치: quality_evaluator 실행 횟수 추적 및 제한
        # 설정에서 max_iterations 사용 (replanner와 동일한 값 공유)
        MAX_QUALITY_EVALUATIONS = _settings.replanner.max_iterations

        # quality_eval_count는 전역 카운터이므로 직접 접근
        metadata = state.get("metadata", {})
        quality_eval_count = metadata.get("quality_eval_count", 0) + 1
        metadata["quality_eval_count"] = quality_eval_count

        if quality_eval_count > MAX_QUALITY_EVALUATIONS:
            logger.warning(f"[should_retry_routing] Maximum quality evaluations ({MAX_QUALITY_EVALUATIONS}) exceeded, forcing termination")
            return "answer"

        # ============================================
        # 1. 글로벌 루프 방지 시스템 초기화
        # ============================================
        loop_prevention = initialize_loop_prevention(metadata)
        current_iteration = loop_prevention["global_iteration_count"]

        logger.info(f"[should_retry_routing] Global iteration: {current_iteration}/25")

        # ============================================
        # 2. 즉시 종료 조건들
        # ============================================

        # 2a. 최대 글로벌 반복 횟수 초과
        if current_iteration >= 25:
            log_forced_termination(
                "Maximum global iterations (25) reached",
                loop_prevention,
                current_iteration
            )
            return "answer"

        # 2b. 평가 실패 시 안전한 종료
        if not evaluator_data.get("success", False):
            logger.warning("[should_retry_routing] Quality evaluation failed, proceeding to answer")
            return "answer"

        # 2c. 루프 패턴 감지
        loop_detected = detect_routing_loop(loop_prevention)
        if loop_detected["is_loop"]:
            log_forced_termination(
                f"Routing loop detected: {loop_detected['pattern']}",
                loop_prevention,
                current_iteration
            )
            return "answer"

        # ============================================
        # 3. 평가 결과 분석
        # ============================================
        cumulative_results = evaluator_data.get("cumulative_evaluation_results", {})
        current_results = evaluator_data.get("evaluation_results", {})

        if not cumulative_results:
            cumulative_results = current_results

        if not cumulative_results:
            logger.warning("[should_retry_routing] No evaluation results found, proceeding to answer")
            return "answer"

        # 실패/성공 쿼리 분석
        failed_queries, successful_queries, total_queries = analyze_query_results(cumulative_results)

        logger.info(f"[should_retry_routing] Query analysis: {len(failed_queries)}/{total_queries} failed")

        # ============================================
        # 4. 라우팅 결정 로직 (루프 방지 적용)
        # ============================================
        iteration_count = replanner_data.get("iteration_count", 0)
        max_iterations = min(10, 25 - current_iteration)  # 글로벌 한계 내에서 조정

        routing_decision = determine_iterative_routing_with_loop_prevention(
            failed_queries,
            successful_queries,
            total_queries,
            iteration_count,
            max_iterations,
            replanner_data,
            loop_prevention,
            current_iteration,
            metadata
        )

        route = routing_decision["route"]

        # ============================================
        # 5. 루프 방지 메타데이터 업데이트
        # ============================================
        update_loop_prevention_metadata(state, route, routing_decision, loop_prevention)

        logger.info(f"[should_retry_routing] Final decision: {route} - {routing_decision['reason']}")

        return route

    except Exception as e:
        logger.error(f"[should_retry_routing] Error during routing decision: {e}")
        return "answer"  # 에러 시 안전하게 answer로


def determine_iterative_routing_with_loop_prevention(
    failed_queries: List[FailedQueryInfo],
    successful_queries: List[str],
    total_queries: int,
    iteration_count: int,
    max_iterations: int,
    replanner_data: Dict[str, Any],
    loop_prevention: LoopPreventionData,
    global_iteration: int,
    metadata: Dict[str, Any]
) -> RoutingDecision:
    """루프 방지가 적용된 반복 라우팅 결정

    Args:
        failed_queries: 실패한 서브쿼리들
        successful_queries: 성공한 서브쿼리 ID들
        total_queries: 전체 서브쿼리 수
        iteration_count: 현재 반복 횟수
        max_iterations: 최대 반복 횟수 (글로벌 제한 적용됨)
        replanner_data: 재계획자 데이터
        loop_prevention: 루프 방지 데이터
        global_iteration: 글로벌 반복 횟수
        metadata: AgentState 메타데이터

    Returns:
        RoutingDecision: 라우팅 결정 정보
    """

    # 1. 모든 쿼리가 성공한 경우 -> answer
    if not failed_queries:
        return {
            "route": "answer",
            "reason": f"All {total_queries} queries passed evaluation",
            "strategy": "success_completion"
        }

    # 4. 글로벌 반복 임계값 완화 - 더 많은 반복 허용
    if global_iteration >= 40:  # 기존 20 -> 40으로 2배 증가
        return {
            "route": "answer",
            "reason": f"Approaching global iteration limit ({global_iteration}/25), forcing termination",
            "strategy": "global_limit_prevention"
        }

    # 5. 로컬 최대 반복 횟수 초과 -> answer
    if iteration_count >= max_iterations:
        return {
            "route": "answer",
            "reason": f"Local maximum iterations ({max_iterations}) reached",
            "strategy": "local_limit_termination"
        }

    # 4. 진동 패턴으로 인한 조기 종료
    routing_history = loop_prevention.get("routing_history", [])
    if len(routing_history) >= 4:
        try:
            oscillation = detect_oscillation_pattern(routing_history[-4:])
            if oscillation["detected"]:
                return {
                    "route": "answer",
                    "reason": f"Oscillation pattern detected: {oscillation['pattern']}",
                    "strategy": "oscillation_prevention"
                }
        except Exception as e:
            logger.warning(f"[routing] Oscillation detection error: {e}")
            # Continue with normal routing logic

    # 6. retrieve 루프 감지 완화 (7회 중 5회 이상 - 더 많은 재시도 허용)
    if len(routing_history) >= 7:
        recent_7 = routing_history[-7:]
        retrieve_count = recent_7.count("retrieve")

        # 7회 중 5회 이상 retrieve면 루프로 간주
        if retrieve_count >= 5:
            logger.warning(f"[routing] Retrieve loop detected: {recent_7}")
            return {
                "route": "answer",
                "reason": f"Retrieve loop detected: {retrieve_count}/7 recent calls were retrieve",
                "strategy": "retrieve_loop_prevention"
            }

    # 7. queries_consumed 후 retrieve 호출 차단
    if replanner_data.get("status") == "queries_consumed":
        if routing_history and routing_history[-1] == "retrieve":
            logger.error("[routing] Invalid routing: retrieve after queries_consumed")
            return {
                "route": "answer",
                "reason": "Invalid routing sequence: retrieve after queries_consumed",
                "strategy": "invalid_routing_prevention"
            }

    # 6. 재구성된 쿼리가 있고 아직 재검색이 안 된 경우 -> retrieve (루프 방지 개선)
    if replanner_data.get("status") == "completed":
        reconstructions = replanner_data.get("reconstructions", [])
        if reconstructions:
            return {
                "route": "retrieve",
                "reason": f"Reconstructed queries ready for re-search: {len(reconstructions)} queries",
                "strategy": "search_reconstructed"
            }
    elif replanner_data.get("status") == "queries_consumed":
        # 재구성된 쿼리가 이미 사용됨 - 새로운 재구성이 있으면 먼저 retrieve
        reconstructions = replanner_data.get("reconstructions", [])
        # 사용되지 않은 reconstruction만 필터링
        unused_reconstructions = [r for r in reconstructions if not r.get("consumed", False)]
        if unused_reconstructions:
            logger.info(f"[routing] New reconstructions available despite consumed status, routing to retrieve")
            return {
                "route": "retrieve",
                "reason": f"New reconstructed queries ready for re-search: {len(unused_reconstructions)} queries",
                "strategy": "search_new_reconstructions"
            }

        # 실패 쿼리가 있으면 재계획 (정상적인 반복 허용)
        failed_count = len(failed_queries)

        if failed_count > 0 and global_iteration < 15:
            # 실패한 서브쿼리에 대해 정상적인 반복 수행
            logger.info(f"[routing] {failed_count}개 실패 쿼리 재계획 (모든 쿼리 성공까지 반복)")
            return {
                "route": "replanner",
                "reason": f"Replanning {failed_count} failed queries until all succeed",
                "strategy": "continue_until_all_succeed"
            }
        else:
            # 모든 쿼리 성공 또는 최대 반복 도달
            if failed_count == 0:
                logger.info(f"[routing] All {len(successful_queries)} queries succeeded!")
            else:
                logger.warning(f"[routing] Max iterations reached: {len(successful_queries)} success, {failed_count} failed")
            return {
                "route": "answer",
                "reason": f"All queries processed: {len(successful_queries)} successful, {failed_count} failed",
                "strategy": "final_completion"
            }

    # 6. 실패 비율과 패턴 분석 (강화된 기준)
    failed_count = len(failed_queries)
    failed_ratio = failed_count / total_queries

    # 7. 재계획 필요 조건들 (더 보수적 적용)

    # 조건 A: 실패 쿼리 존재 시 재계획 (1개라도 실패하면 재계획)
    if failed_count >= 1 and global_iteration < 15:
        return {
            "route": "replanner",
            "reason": f"Failed queries detected: {failed_count}/{total_queries} queries need replanning",
            "strategy": "regenerate_failed_queries"
        }

    # 8. 최종 결정: 실패 쿼리가 있으면 replanner, 없으면 answer
    if failed_count > 0:
        if global_iteration >= 15:
            logger.warning(f"[routing] Maximum iterations reached but {failed_count} queries still failed")
            return {
                "route": "answer",
                "reason": f"Maximum iterations (15) reached with {failed_count}/{total_queries} failures",
                "strategy": "forced_termination_with_failures"
            }
        else:
            return {
                "route": "replanner",
                "reason": f"Continuing replanning: {failed_count}/{total_queries} queries still need improvement",
                "strategy": "continue_until_all_success"
            }
    else:
        return {
            "route": "answer",
            "reason": f"All queries successful: 0/{total_queries} failures",
            "strategy": "all_queries_successful"
        }


def determine_iterative_routing(
    failed_queries: List[FailedQueryInfo],
    successful_queries: List[str],
    total_queries: int,
    iteration_count: int,
    max_iterations: int,
    replanner_data: Dict[str, Any]
) -> RoutingDecision:
    """반복적 개선을 위한 라우팅 결정 (루프 방지 미적용 버전)

    Args:
        failed_queries: 실패한 서브쿼리들
        successful_queries: 성공한 서브쿼리 ID들
        total_queries: 전체 서브쿼리 수
        iteration_count: 현재 반복 횟수
        max_iterations: 최대 반복 횟수
        replanner_data: 재계획자 데이터

    Returns:
        RoutingDecision: 라우팅 결정 정보
    """

    # 1. 모든 쿼리가 성공한 경우 -> answer
    if not failed_queries:
        return {
            "route": "answer",
            "reason": f"All {total_queries} queries passed evaluation",
            "strategy": None
        }

    # 2. 최대 반복 횟수 초과 -> answer (강제 종료)
    if iteration_count >= max_iterations:
        return {
            "route": "answer",
            "reason": f"Maximum iterations ({max_iterations}) reached, proceeding with available results",
            "strategy": "forced_termination"
        }

    # 3. 재구성된 쿼리가 있고 아직 재검색이 안 된 경우 -> retrieve (루프 방지 개선)
    if replanner_data.get("status") == "completed":
        reconstructions = replanner_data.get("reconstructions", [])
        if reconstructions:
            return {
                "route": "retrieve",
                "reason": f"Reconstructed queries ready for re-search: {len(reconstructions)} queries",
                "strategy": "search_reconstructed"
            }
    elif replanner_data.get("status") == "queries_consumed":
        # 재구성된 쿼리가 이미 사용됨 - 더 이상 재계획 불가, answer로 진행
        logger.info("[routing] Queries already consumed, proceeding to answer")
        return {
            "route": "answer",
            "reason": "Reconstructed queries have been consumed, no further replanning allowed",
            "strategy": "finalize_with_available_results"
        }

    # 4. 실패 비율과 패턴 분석
    failed_count = len(failed_queries)
    failed_ratio = failed_count / total_queries

    # 5. 재계획 필요 조건들

    # 조건 A: 높은 실패 비율 (50% 이상)
    if failed_ratio >= 0.5 and failed_count >= 2:
        return {
            "route": "replanner",
            "reason": f"High failure ratio: {failed_ratio:.1%} ({failed_count}/{total_queries})",
            "strategy": "regenerate_high_failure"
        }

    # 조건 B: 핵심 앵커 쿼리(Q1) 실패
    anchor_failed = any(
        q.get("query_id", "").startswith("Q1") for q in failed_queries
    )
    if anchor_failed:
        return {
            "route": "replanner",
            "reason": "Critical anchor query (Q1) failed - requires reconstruction",
            "strategy": "regenerate_anchor_focused"
        }

    # 조건 C: 낮은 신뢰도의 실패들 (재평가 가능성)
    low_confidence_failures = [
        q for q in failed_queries
        if q.get("confidence", 1.0) < 0.6
    ]
    if len(low_confidence_failures) >= 2:
        return {
            "route": "replanner",
            "reason": f"Low confidence in {len(low_confidence_failures)} failed evaluations",
            "strategy": "regenerate_low_confidence"
        }

    # 조건 D: 연속 실패 패턴 (같은 쿼리가 여러 번 실패)
    iteration_history = replanner_data.get("iteration_history", [])
    if len(iteration_history) >= 2:
        # 최근 2회 반복에서 같은 쿼리가 계속 실패하는지 확인
        persistent_failures = identify_persistent_failures(failed_queries, iteration_history)
        if persistent_failures and failed_count >= 1:
            return {
                "route": "replanner",
                "reason": f"Persistent failures detected: {len(persistent_failures)} queries",
                "strategy": "regenerate_persistent_failures"
            }

    # 6. 낮은 실패 비율이지만 개선 여지가 있는 경우
    if failed_count > 0 and failed_ratio < 0.5:
        # 마지막 시도: 한 번 더 재구성
        if iteration_count < max_iterations - 1:  # 마지막 반복이 아닌 경우
            return {
                "route": "replanner",
                "reason": f"Minor failures ({failed_count} queries) but room for improvement",
                "strategy": "optimize_minor_failures"
            }
        else:
            # 마지막 반복이면 그냥 종료
            return {
                "route": "answer",
                "reason": f"Minor failures acceptable in final iteration: {failed_ratio:.1%}",
                "strategy": "accept_minor_failures"
            }

    # 7. 기본값: answer (안전한 종료)
    return {
        "route": "answer",
        "reason": f"Default termination with {failed_count} failures",
        "strategy": "default_termination"
    }


# ============================================
# Backward Compatibility - 기존 private 함수명 유지
# ============================================

# 기존 코드와의 호환성을 위해 private 함수명으로 alias 제공
_analyze_query_results = analyze_query_results
_identify_persistent_failures = identify_persistent_failures
_determine_iterative_routing_with_loop_prevention = determine_iterative_routing_with_loop_prevention
_determine_iterative_routing = determine_iterative_routing
