"""
Quality Evaluator Node - LangGraph Integration Module
=====================================================

LORIN 시스템의 LangGraph workflow에 통합되는 품질 평가 노드입니다.
FAISS 검색 결과를 EXAONE 모델로 평가하여 관련성을 판단합니다.

주요 기능:
- state.metadata.faiss_retriever 데이터 처리
- 서브쿼리별 관련성 평가 실행
- state.metadata.quality_evaluator에 결과 저장
- 기존 state 및 metadata 보존
- 상세한 평가 통계 및 성능 메트릭 제공

통합 포인트:
- 입력: FAISS retriever 결과
- 출력: 평가된 관련성 정보
- 의존성: quality_evaluator.py, chatbot.py
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage
from ...make_faiss.search_engine import SearchResult
from ...logger.logger import get_logger
from ...config.settings import get_settings
from ..state import AgentState
from .quality_evaluator import (
    QualityEvaluator,
    QueryEvaluationResult,
    EvaluationStats,
    DocumentEvaluation
)

# ── 라우팅 결정 및 루프 방지 모듈 임포트 ──────────────────────
from ..routing.routing_decision import (
    should_retry_routing,
    analyze_query_results as _analyze_query_results,
    identify_persistent_failures as _identify_persistent_failures,
    determine_iterative_routing_with_loop_prevention as _determine_iterative_routing_with_loop_prevention,
    determine_iterative_routing as _determine_iterative_routing,
)
from ..routing.loop_prevention import (
    initialize_loop_prevention as _initialize_loop_prevention,
    detect_routing_loop as _detect_routing_loop,
    detect_oscillation_pattern as _detect_oscillation_pattern,
    detect_consecutive_pattern as _detect_consecutive_pattern,
    detect_cycle_pattern as _detect_cycle_pattern,
    log_forced_termination as _log_forced_termination,
    update_loop_prevention_metadata as _update_loop_prevention_metadata,
)
from ..schema import MetadataManager

# DataClasses are now imported from quality_evaluator module

logger = get_logger(__name__)

# ── 중앙 집중식 설정 로드 ───────────────────────────────────
_settings = get_settings()
_qe_cfg = _settings.quality_evaluator

# ─────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────

def _build_query_success_timeline(
    existing_timeline: Dict[str, Any],
    current_results: Dict[str, Any],
    current_iteration: int,
    current_timestamp: str,
    search_timestamp: str
) -> Dict[str, Any]:
    """쿼리별 성공 시점 추적 시스템 구축

    Args:
        existing_timeline: 기존 성공 시점 추적 데이터
        current_results: 현재 평가 결과
        current_iteration: 현재 반복 횟수
        current_timestamp: 현재 평가 시각
        search_timestamp: 검색 실행 시각

    Returns:
        Dict[str, Any]: 업데이트된 성공 시점 추적 데이터
    """
    timeline = existing_timeline.copy() if existing_timeline else {}

    for qid, result_data in current_results.items():
        is_relevant = result_data.get("is_relevant", False)

        # 쿼리가 성공했고, 아직 성공 기록이 없는 경우만 최초 성공 시점 기록
        if is_relevant and qid not in timeline:
            timeline[qid] = {
                "iteration": current_iteration,
                "success_timestamp": current_timestamp,
                "search_timestamp": search_timestamp,
                "relevance_score": result_data.get("relevance_score", 0.0),
                "confidence": result_data.get("confidence", 0.0),
                "first_success": True
            }
            logger.info(f"[_build_query_success_timeline] First success recorded for {qid} at iteration {current_iteration}")
        elif is_relevant and qid in timeline:
            # 이미 성공한 쿼리의 재평가 시 latest 정보만 업데이트 (first_success는 유지)
            timeline[qid].update({
                "latest_iteration": current_iteration,
                "latest_success_timestamp": current_timestamp,
                "latest_search_timestamp": search_timestamp,
                "latest_relevance_score": result_data.get("relevance_score", 0.0),
                "latest_confidence": result_data.get("confidence", 0.0)
            })
            logger.info(f"[_build_query_success_timeline] Updated latest success for {qid} at iteration {current_iteration}")

    return timeline

def _build_query_status_by_iteration(
    existing_status: Dict[str, Dict[str, str]],
    current_results: Dict[str, Any],
    current_iteration: int
) -> Dict[str, Dict[str, str]]:
    """반복별 쿼리 상태 이력 구축

    Args:
        existing_status: 기존 반복별 상태 데이터
        current_results: 현재 평가 결과
        current_iteration: 현재 반복 횟수

    Returns:
        Dict[str, Dict[str, str]]: 업데이트된 반복별 상태 데이터
    """
    status_by_iteration = existing_status.copy() if existing_status else {}

    current_status = {}
    for qid, result_data in current_results.items():
        is_relevant = result_data.get("is_relevant", False)
        current_status[qid] = "pass" if is_relevant else "fail"

    status_by_iteration[str(current_iteration)] = current_status

    logger.info(f"[_build_query_status_by_iteration] Recorded status for iteration {current_iteration}: {current_status}")
    return status_by_iteration

def _get_search_timestamps_from_state(state: AgentState) -> Tuple[str, str]:
    """State에서 검색 실행 시각 추출

    Args:
        state: Current agent state

    Returns:
        Tuple[str, str]: (검색 실행 시각, 현재 시각)
    """
    metadata = state.get("metadata", {})

    # faiss_retriever의 timestamp 사용
    faiss_data = metadata.get("faiss_retriever", {})
    search_timestamp = faiss_data.get("timestamp", "")

    current_timestamp = datetime.now(timezone.utc).isoformat()

    return search_timestamp, current_timestamp

def _reconstruct_search_results(results_data: List[Dict[str, Any]]) -> List[SearchResult]:
    """직렬화된 SearchResult 데이터를 객체로 복원"""
    search_results = []

    for result_dict in results_data:
        search_result = SearchResult(
            rank=result_dict.get("rank", 0),
            score=result_dict.get("score", 0.0),
            content=result_dict.get("content", ""),
            metadata=result_dict.get("metadata", {}),
            line_range=result_dict.get("line_range", []),
            time_start=result_dict.get("time_start", ""),
            time_end=result_dict.get("time_end", "")
        )
        search_results.append(search_result)

    return search_results

def _extract_retrieval_data(state: AgentState) -> Tuple[List[Dict[str, Any]], Dict[str, List[SearchResult]]]:
    """State에서 FAISS 검색 데이터 추출"""
    metadata = state.get("metadata", {})
    faiss_data = metadata.get("faiss_retriever", {})

    if not faiss_data.get("success", False):
        logger.warning("[quality_evaluator_node] FAISS retrieval data not found or unsuccessful")
        return [], {}

    # 서브쿼리 정보 추출
    planner_data = metadata.get("planner", {})
    plan_json = planner_data.get("last_plan_json", {})
    subqueries = plan_json.get("subqueries", [])

    # 디버깅: Planner 서브쿼리 확인
    logger.info(f"🔍 [_extract_retrieval_data] Planner subqueries extraction:")
    logger.info(f"  - Total subqueries: {len(subqueries)}")
    for i, sq in enumerate(subqueries):
        query_id = sq.get("id", "unknown")
        query_text = sq.get("text", "")
        logger.info(f"  - [{query_id}] text='{query_text[:80]}...' (length: {len(query_text)})")

    if not subqueries:
        logger.warning("[quality_evaluator_node] No subqueries found in planner data")
        return [], {}

    # 검색 결과 추출 및 복원
    results_by_qid = faiss_data.get("results_by_qid", {})
    documents_by_query = {}

    for qid, result_data in results_by_qid.items():
        raw_results = result_data.get("results", [])
        documents_by_query[qid] = _reconstruct_search_results(raw_results)

    logger.info(f"[quality_evaluator_node] Extracted {len(subqueries)} subqueries and "
               f"{len(documents_by_query)} result sets")

    return subqueries, documents_by_query

def _prepare_evaluation_batch(
    subqueries: List[Dict[str, Any]],
    documents_by_query: Dict[str, List[SearchResult]]
) -> List[Tuple[Dict[str, Any], List[SearchResult]]]:
    """평가용 배치 데이터 준비"""
    batch_data = []

    for subquery in subqueries:
        query_id = subquery.get("id", "unknown")
        documents = documents_by_query.get(query_id, [])

        if documents:  # 문서가 있는 쿼리만 평가
            batch_data.append((subquery, documents))
        else:
            logger.warning(f"[quality_evaluator_node] No documents found for query {query_id}")

    return batch_data

def _create_evaluation_summary(
    evaluation_results: Dict[str, QueryEvaluationResult],
    stats: EvaluationStats
) -> str:
    """평가 결과 요약 생성"""
    if not evaluation_results:
        return "Quality Evaluation: No new queries evaluated (all previously successful)"

    relevant_queries = sum(1 for result in evaluation_results.values() if result.is_relevant)
    total_queries = len(evaluation_results)

    avg_score = stats.avg_relevance_score
    avg_confidence = stats.avg_confidence

    summary_lines = [
        "Quality Evaluation Complete",
        f"Relevant Queries: {relevant_queries}/{total_queries} ({relevant_queries/total_queries*100:.1f}%)",
        f"Average Relevance Score: {avg_score:.3f}",
        f"Average Confidence: {avg_confidence:.3f}",
        f"Total Documents Evaluated: {stats.total_documents_evaluated}",
        f"Relevant Documents Found: {stats.total_relevant_documents}",
        f"Evaluation Time: {stats.evaluation_time:.2f}s",
        f"Tokens Used: {stats.total_tokens_used}",
        f"API Calls: {stats.api_calls_made}",
        f"Cache Hits: {stats.cache_hits}"
    ]

    if stats.fallback_count > 0:
        summary_lines.append(f"Fallback Evaluations: {stats.fallback_count}")

    return "\n".join(summary_lines)

# ─────────────────────────────────────────────────────────
# Main Quality Evaluator Node
# ─────────────────────────────────────────────────────────

def quality_evaluator_node(chatbot):
    """Factory function for LangGraph-compatible quality evaluation node

    Args:
        chatbot: 공유 Chatbot 인스턴스 (Together API EXAONE 사용)

    Returns:
        Async function that processes AgentState
    """

    async def _quality_evaluator_node_impl(state: AgentState) -> AgentState:
        """LangGraph-compatible quality evaluation node

        이 함수는:
        1. state.metadata.faiss_retriever에서 검색 결과 추출
        2. 각 서브쿼리와 문서들의 관련성을 EXAONE으로 평가
        3. 평가 결과를 state.metadata.quality_evaluator에 저장
        4. 모든 기존 state와 metadata 보존
        5. 평가 요약 메시지를 대화 히스토리에 추가

        Args:
            state: Current agent state from LangGraph

        Returns:
            AgentState: Updated state with quality evaluation results
        """
        logger.info("[quality_evaluator_node] Starting quality evaluation processing")

        # 기존 데이터 추출
        messages: List[BaseMessage] = state.get("messages", [])
        metadata: Dict[str, Any] = state.get("metadata", {})

        try:
            # FAISS 검색 데이터 추출
            subqueries, documents_by_query = _extract_retrieval_data(state)

            if not subqueries or not documents_by_query:
                logger.warning("[quality_evaluator_node] No data available for evaluation")
                return _create_error_state(
                    state, messages, metadata,
                    "No retrieval data available for quality evaluation"
                )

            # 🔧 선택적 재평가: 기존 성공한 쿼리는 재평가하지 않음
            # MetadataManager를 통한 안전한 메타데이터 접근
            existing_evaluator_data = MetadataManager.safe_get_metadata(state, "quality_evaluator") or {}
            existing_cumulative = existing_evaluator_data.get("cumulative_evaluation_results", {})

            # 이미 성공한 쿼리 식별
            previously_successful_queries = set()
            for qid, result_data in existing_cumulative.items():
                if result_data.get("is_relevant", False):
                    previously_successful_queries.add(qid)
            logger.info(f"[quality_evaluator_node] Previously successful queries: {len(previously_successful_queries)} - {list(previously_successful_queries)}")

            # 재평가가 필요한 쿼리만 필터링
            queries_to_evaluate = []
            documents_to_evaluate = {}

            logger.info(f"[quality_evaluator_node] Debug - subqueries type: {type(subqueries)}, length: {len(subqueries)}")
            for i, query_dict in enumerate(subqueries):
                logger.info(f"[quality_evaluator_node] Debug - subquery {i}: {query_dict}")
                qid = query_dict.get("id", "")  # 🔧 수정: 'subquery_id' → 'id'
                logger.info(f"[quality_evaluator_node] Debug - extracted QID: '{qid}'")

                if qid and qid not in previously_successful_queries:
                    queries_to_evaluate.append(query_dict)
                    if qid in documents_by_query:
                        documents_to_evaluate[qid] = documents_by_query[qid]
                        logger.info(f"[quality_evaluator_node] Debug - Added QID {qid} to evaluation")
                    else:
                        logger.warning(f"[quality_evaluator_node] Debug - No documents for QID {qid}")
                else:
                    logger.info(f"[quality_evaluator_node] Debug - Skipped QID {qid} (empty or previously successful)")

            if queries_to_evaluate:
                query_ids = [q.get("id", "") for q in queries_to_evaluate]  # 🔧 수정: 'subquery_id' → 'id'
                logger.info(f"[quality_evaluator_node] Queries requiring evaluation: {len(queries_to_evaluate)} - {query_ids}")
                # 선별적 평가 배치 준비
                evaluation_batch = _prepare_evaluation_batch(queries_to_evaluate, documents_to_evaluate)
            else:
                logger.info("[quality_evaluator_node] All queries previously successful, skipping evaluation")
                evaluation_batch = []

            # 평가할 쿼리가 있는 경우에만 실제 평가 수행
            if evaluation_batch:
                logger.info(f"[quality_evaluator_node] Prepared {len(evaluation_batch)} queries for evaluation")

                # 🔧 GPU 메모리 기반 동적 동시성 제한
                max_concurrent = 1  # 기본값: 보수적 순차 실행
                try:
                    import torch
                    if torch.cuda.is_available():
                        # GPU 0만 사용 (EXAONE 전용)
                        gpu0_free = torch.cuda.mem_get_info(0)[0] / 1e9
                        gpu0_allocated = torch.cuda.memory_allocated(0) / 1e9

                        # 보수적인 동시성 정책 (긴 입력 고려)
                        if gpu0_free > 10.0:
                            max_concurrent = 5  # 10GB 이상 여유: 최대 5개 병렬
                        elif gpu0_free > 6.0:
                            max_concurrent = 3  # 6GB 이상 여유: 3개 병렬
                        else:
                            max_concurrent = 2  # 그 외: 2개 병렬

                        logger.info(
                            f"[quality_evaluator_node] 💾 GPU 0: Free={gpu0_free:.2f}GB, "
                            f"Allocated={gpu0_allocated:.2f}GB → max_concurrent={max_concurrent}"
                        )
                except ImportError:
                    logger.warning("[quality_evaluator_node] torch not available, using max_concurrent=1")
                    max_concurrent = 1

                # Quality Evaluator 초기화 (공유 chatbot 사용 - 로컬 EXAONE)
                evaluator = QualityEvaluator(
                    chatbot=chatbot,  # 공유 Chatbot 인스턴스 사용 (GPU 메모리 절약)
                    relevance_threshold=0.4,  # 0.5 → 0.4: 프롬프트 "Moderate relevance" 철학과 일치
                    confidence_threshold=0.6,
                    max_docs_per_query=3,  # 5 → 3: 프롬프트 크기 감소
                    max_retries=3,
                    timeout_seconds=120.0  # 30초 → 120초: Multi-GPU 분산 고려
                )

                # 배치 평가 실행 (GPU 메모리 기반 동적 병렬 처리)
                logger.info(f"[quality_evaluator_node] Starting batch evaluation with max_concurrent={max_concurrent}")
                evaluation_results, evaluation_stats = await evaluator.evaluate_batch_queries(
                    evaluation_batch,
                    max_concurrent=max_concurrent  # 동적으로 결정된 동시성 제한
                )
            else:
                # 모든 쿼리가 이미 성공한 경우 - 빈 결과로 진행
                logger.info("[quality_evaluator_node] No new queries to evaluate - all previously successful")
                evaluation_results = {}
                # 빈 evaluation_stats 생성 (None 대신 기본값 사용)
                from .quality_evaluator import EvaluationStats
                evaluation_stats = EvaluationStats(
                    total_queries=0,
                    total_documents_evaluated=0,
                    total_relevant_documents=0,
                    avg_relevance_score=0.0,
                    avg_confidence=0.0,
                    evaluation_time=0.0,
                    total_tokens_used=0,
                    api_calls_made=0,
                    cache_hits=0,
                    fallback_count=0
                )

            # 기존 평가 결과와 병합 (누적 평가 지원)
            existing_evaluator_data = metadata.get("quality_evaluator", {})
            existing_results = existing_evaluator_data.get("evaluation_results", {})
            existing_cumulative = existing_evaluator_data.get("cumulative_evaluation_results", {})

            # 현재 평가 결과 직렬화 (replanner 호환성을 위한 필드 매핑 포함)
            current_serializable_results = {}
            for qid, result in evaluation_results.items():
                current_serializable_results[qid] = {
                    # 새로운 구조
                    "query_id": result.query_id,
                    "query_text": result.query_text,
                    "is_relevant": result.is_relevant,
                    "relevance_score": result.relevance_score,
                    "document_evaluations": [asdict(doc_eval) for doc_eval in result.document_evaluations],
                    "evaluation_time": result.evaluation_time,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "total_documents": result.total_documents,
                    "relevant_documents": result.relevant_documents,
                    "avg_doc_score": result.avg_doc_score,
                    "max_doc_score": result.max_doc_score,
                    "tokens_used": result.tokens_used,
                    "error": result.error,
                    "fallback_used": result.fallback_used,
                    "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "iteration_count": metadata.get("replanner", {}).get("iteration_count", 0),

                    # 🔧 replanner 호환성을 위한 backward compatibility 필드
                    "confidence_score": result.confidence,  # replanner가 기대하는 필드명
                    "evaluation_details": {  # replanner에게 상세 피드백 전달
                        "reasoning": result.reasoning,
                        "improvement_suggestions": result.improvement_suggestions,
                            "query_effectiveness": result.query_effectiveness
                        }
                    }

            # 누적 결과 생성: 기존 성공 결과 보존 + 새로운 평가 결과 추가
            cumulative_results = _merge_evaluation_results(
            existing_cumulative or existing_results,
            current_serializable_results,
            metadata.get("replanner", {})
            )

            # 🔧 중복 쿼리 필터링: 95% 이상 유사한 쿼리 중 relevance_score가 낮은 것을 제거
            cumulative_results = _filter_duplicate_queries(cumulative_results)

            # 현재 반복의 쿼리별 관련성 요약 생성
            current_relevance_summary = {}
            for qid, result in evaluation_results.items():
                current_relevance_summary[qid] = {
                    "query_text": result.query_text,
                    "is_relevant": result.is_relevant,
                    "relevance_score": result.relevance_score,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning
                }

            # 누적 관련성 요약 생성
            cumulative_relevance_summary = {}
            for qid, result_data in cumulative_results.items():
                cumulative_relevance_summary[qid] = {
                    "query_text": result_data.get("query_text", ""),
                    "is_relevant": result_data.get("is_relevant", False),
                    "relevance_score": result_data.get("relevance_score", 0.0),
                    "confidence": result_data.get("confidence", 0.0),
                    "reasoning": result_data.get("reasoning", ""),
                    "evaluation_timestamp": result_data.get("evaluation_timestamp", ""),
                    "iteration_count": result_data.get("iteration_count", 0)
                }

            # ============================================
            # 새로운 쿼리 성공 시점 추적 시스템 구축
            # ============================================
            current_iteration = metadata.get("replanner", {}).get("iteration_count", 0)
            search_timestamp, current_timestamp = _get_search_timestamps_from_state(state)

            # 기존 추적 데이터 로드
            existing_evaluator_data = metadata.get("quality_evaluator", {})
            existing_timeline = existing_evaluator_data.get("query_success_timeline", {})
            existing_status_history = existing_evaluator_data.get("query_status_by_iteration", {})

            # 쿼리 성공 시점 추적 업데이트
            query_success_timeline = _build_query_success_timeline(
            existing_timeline,
            current_serializable_results,
            current_iteration,
            current_timestamp,
            search_timestamp
            )

            # 반복별 쿼리 상태 이력 업데이트
            query_status_by_iteration = _build_query_status_by_iteration(
            existing_status_history,
            current_serializable_results,
            current_iteration
            )

            # 성공 통계 계산
            first_success_queries = [qid for qid, info in query_success_timeline.items()
                           if info.get("first_success", False) and info.get("iteration") == current_iteration]
            total_successful_queries = len([qid for qid, info in query_success_timeline.items()])

            logger.info(f"[quality_evaluator_node] Success tracking: {len(first_success_queries)} first successes, {total_successful_queries} total successful")

            # metadata.quality_evaluator에 저장 (누적 지원 + 새로운 추적 시스템)
            quality_evaluator_metadata = {
            "evaluation_results": current_serializable_results,  # 현재 반복 결과
            "cumulative_evaluation_results": cumulative_results,  # 누적 결과
            "relevance_summary": current_relevance_summary,  # 현재 반복 요약
            "cumulative_relevance_summary": cumulative_relevance_summary,  # 누적 요약
            "query_success_timeline": query_success_timeline,  # 🆕 쿼리별 성공 시점 추적
            "query_status_by_iteration": query_status_by_iteration,  # 🆕 반복별 상태 이력
            "success_statistics": {  # 🆕 성공 통계
            "first_success_this_iteration": len(first_success_queries),
            "total_successful_queries": total_successful_queries,
            "first_success_query_ids": first_success_queries,
            "current_iteration": current_iteration
            },
            "evaluation_stats": asdict(evaluation_stats),
            "config": {
            "relevance_threshold": evaluator.relevance_threshold,
            "confidence_threshold": evaluator.confidence_threshold,
            "max_docs_per_query": evaluator.max_docs_per_query,
            "model": "lgai/exaone-3-5-32b-instruct"
            },
            "evaluation_mode": "cumulative" if existing_cumulative or existing_results else "initial",
            "iteration_count": current_iteration,
            "timestamp": current_timestamp,
            "search_timestamp": search_timestamp,  # 🆕 검색 시각 기록
            "success": True
            }

            # 메타데이터 업데이트 (기존 데이터 보존)
            new_metadata = metadata.copy()
            new_metadata["quality_evaluator"] = quality_evaluator_metadata
            new_metadata["last_agent"] = "quality_evaluator"

            # 요약 메시지 생성
            summary_content = _create_evaluation_summary(evaluation_results, evaluation_stats)

            evaluation_message = AIMessage(
            content=summary_content,
            additional_kwargs={
            "agent_name": "quality_evaluator",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_stats": asdict(evaluation_stats),
            "relevant_queries": sum(1 for r in evaluation_results.values() if r.is_relevant),
            "total_queries": len(evaluation_results)
            }
            )

            new_messages = messages + [evaluation_message]

            logger.info(f"[quality_evaluator_node] Successfully evaluated {len(evaluation_results)} queries")
            logger.info(f"[quality_evaluator_node] Relevant queries: "
               f"{sum(1 for r in evaluation_results.values() if r.is_relevant)}/{len(evaluation_results)}")
            logger.info(f"[quality_evaluator_node] Average relevance score: {evaluation_stats.avg_relevance_score:.3f}")

            # 업데이트된 state 반환
            return type(state)(
            messages=new_messages,
            context_messages=state.get("context_messages", []),
            metadata=new_metadata,
            current_agent="quality_evaluator",
            session_id=state.get("session_id", "")
            )

        except Exception as e:
            error_msg = f"Quality evaluation failed: {str(e)}"
            logger.error(f"[quality_evaluator_node] {error_msg}")
            return _create_error_state(state, messages, metadata, error_msg)

    return _quality_evaluator_node_impl

def _create_error_state(
    state: AgentState,
    messages: List[BaseMessage],
    metadata: Dict[str, Any],
    error_msg: str
) -> AgentState:
    """에러 상태 생성 헬퍼 함수"""

    # 에러 메타데이터 저장
    new_metadata = metadata.copy()
    new_metadata["quality_evaluator"] = {
        "success": False,
        "error": error_msg,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    new_metadata["last_agent"] = "quality_evaluator"

    # 에러 메시지 생성
    error_message = AIMessage(
        content=f"Quality Evaluation Error: {error_msg}",
        additional_kwargs={
            "agent_name": "quality_evaluator",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": True
        }
    )

    new_messages = messages + [error_message]

    return type(state)(
        messages=new_messages,
        context_messages=state.get("context_messages", []),
        metadata=new_metadata,
        current_agent="quality_evaluator",
        session_id=state.get("session_id", "")
    )

# ─────────────────────────────────────────────────────────
# Utility Functions for Result Access
# ─────────────────────────────────────────────────────────

def get_evaluation_results_for_query(state: AgentState, query_id: str) -> Optional[QueryEvaluationResult]:
    """특정 쿼리의 평가 결과 추출

    Args:
        state: Agent state containing evaluation results
        query_id: Query identifier to look up

    Returns:
        Optional[QueryEvaluationResult]: Evaluation result or None if not found
    """
    metadata = state.get("metadata", {})
    evaluator_data = metadata.get("quality_evaluator", {})

    if not evaluator_data.get("success", False):
        return None

    evaluation_results = evaluator_data.get("evaluation_results", {})
    result_data = evaluation_results.get(query_id)

    if not result_data:
        return None

    # DocumentEvaluation 객체 복원
    doc_evaluations = []
    for doc_data in result_data.get("document_evaluations", []):
        doc_eval = DocumentEvaluation(**doc_data)
        doc_evaluations.append(doc_eval)

    # QueryEvaluationResult 객체 복원
    return QueryEvaluationResult(
        query_id=result_data["query_id"],
        query_text=result_data["query_text"],
        is_relevant=result_data["is_relevant"],
        relevance_score=result_data["relevance_score"],
        document_evaluations=doc_evaluations,
        evaluation_time=result_data["evaluation_time"],
        confidence=result_data["confidence"],
        reasoning=result_data["reasoning"],
        total_documents=result_data["total_documents"],
        relevant_documents=result_data["relevant_documents"],
        avg_doc_score=result_data["avg_doc_score"],
        max_doc_score=result_data["max_doc_score"],
        tokens_used=result_data["tokens_used"],
        error=result_data.get("error"),
        fallback_used=result_data.get("fallback_used", False)
    )

def get_relevance_summary(state: AgentState) -> Dict[str, Dict[str, Any]]:
    """모든 쿼리의 관련성 요약 정보 추출

    Args:
        state: Agent state containing evaluation results

    Returns:
        Dict[str, Dict[str, Any]]: Query ID별 관련성 요약
    """
    # MetadataManager를 통한 안전한 메타데이터 접근
    evaluator_data = MetadataManager.safe_get_metadata(state, "quality_evaluator") or {}

    if not evaluator_data.get("success", False):
        return {}

    return evaluator_data.get("relevance_summary", {})

def format_evaluation_results(evaluator_metadata: Dict[str, Any]) -> str:
    """평가 결과를 사람이 읽기 좋은 형태로 포맷팅

    Args:
        evaluator_metadata: Metadata from quality_evaluator namespace

    Returns:
        str: Formatted evaluation results string
    """
    if not evaluator_metadata.get("success", False):
        return f"Quality evaluation failed: {evaluator_metadata.get('error', 'Unknown error')}"

    evaluation_results = evaluator_metadata.get("evaluation_results", {})
    evaluation_stats = evaluator_metadata.get("evaluation_stats", {})

    lines = ["=== Quality Evaluation Results ===\n"]

    # 전체 통계
    relevant_count = sum(1 for result in evaluation_results.values()
                        if result.get("is_relevant", False))
    total_count = len(evaluation_results)

    lines.append(f"Overall Statistics:")
    lines.append(f"  Relevant Queries: {relevant_count}/{total_count} ({relevant_count/total_count*100:.1f}%)")
    lines.append(f"  Average Relevance Score: {evaluation_stats.get('avg_relevance_score', 0.0):.3f}")
    lines.append(f"  Average Confidence: {evaluation_stats.get('avg_confidence', 0.0):.3f}")
    lines.append(f"  Evaluation Time: {evaluation_stats.get('evaluation_time', 0.0):.2f}s")
    lines.append("")

    # 쿼리별 결과
    for qid, result_data in evaluation_results.items():
        query_text = result_data.get("query_text", "")
        is_relevant = result_data.get("is_relevant", False)
        relevance_score = result_data.get("relevance_score", 0.0)
        confidence = result_data.get("confidence", 0.0)
        reasoning = result_data.get("reasoning", "")

        status_emoji = "✅" if is_relevant else "❌"
        lines.append(f"{status_emoji} Query {qid}: {query_text}")
        lines.append(f"  Relevance: {relevance_score:.3f} (Confidence: {confidence:.3f})")
        lines.append(f"  Reasoning: {reasoning}")
        lines.append("")

    # 성능 메트릭
    lines.append("=== Performance Metrics ===")
    lines.append(f"Total Tokens Used: {evaluation_stats.get('total_tokens_used', 0)}")
    lines.append(f"API Calls Made: {evaluation_stats.get('api_calls_made', 0)}")
    lines.append(f"Cache Hits: {evaluation_stats.get('cache_hits', 0)}")
    if evaluation_stats.get('fallback_count', 0) > 0:
        lines.append(f"Fallback Evaluations: {evaluation_stats.get('fallback_count', 0)}")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────
# Retry Routing Logic - 별도 모듈에서 임포트됨
# (routing_decision.py, loop_prevention.py 참조)
# ─────────────────────────────────────────────────────────

def _update_retry_metadata(state: AgentState, retry_info: Dict[str, Any]) -> None:
    """재시도 메타데이터 업데이트 (상태 변경 없이 정보만 저장)"""

    # Note: 이 함수는 state를 직접 수정하지 않고,
    # 정보만 로깅합니다. 실제 state 업데이트는 route에서 처리합니다.

    logger.info(f"[_update_retry_metadata] Retry metadata: {retry_info}")

    # 향후 필요시 temporary storage나 logging 확장 가능
    pass

def _ensure_backward_compatibility_fields(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """평가 결과에 replanner 호환성 필드가 있는지 확인하고 추가

    Args:
        result_data: 평가 결과 데이터

    Returns:
        Dict[str, Any]: 호환성 필드가 추가된 결과 데이터
    """
    if not result_data:
        return result_data

    # 기존 결과를 복사
    enhanced_result = result_data.copy()

    # backward compatibility 필드가 없으면 추가
    if "confidence_score" not in enhanced_result:
        enhanced_result["confidence_score"] = enhanced_result.get("confidence", 0.0)

    if "evaluation_details" not in enhanced_result:
        # 새 형식: 딕셔너리로 상세 정보 제공
        enhanced_result["evaluation_details"] = {
            "reasoning": enhanced_result.get("reasoning", ""),
            "improvement_suggestions": enhanced_result.get("improvement_suggestions", ""),
            "query_effectiveness": enhanced_result.get("query_effectiveness", "")
        }

    return enhanced_result

def _merge_evaluation_results(
    existing_results: Dict[str, Any],
    new_results: Dict[str, Any],
    replanner_data: Dict[str, Any]
) -> Dict[str, Any]:
    """기존 평가 결과와 새로운 평가 결과를 병합 (성공 보존 원칙)

    Args:
        existing_results: 기존 평가 결과
        new_results: 새로운 평가 결과
        replanner_data: 재계획자 정보

    Returns:
        Dict[str, Any]: 병합된 누적 평가 결과
    """

    if not existing_results:
        return new_results.copy()

    if not new_results:
        # 🔧 기존 결과에도 호환성 필드 추가
        return {qid: _ensure_backward_compatibility_fields(result)
                for qid, result in existing_results.items()}

    # 🔧 기존 결과에 호환성 필드 추가
    merged_results = {qid: _ensure_backward_compatibility_fields(result)
                     for qid, result in existing_results.items()}

    # 재구성된 쿼리 ID 목록 (이들은 새로운 평가 결과로 업데이트)
    reconstructed_query_ids = set()
    if replanner_data.get("status") in ["completed", "queries_consumed"]:
        reconstructions = replanner_data.get("reconstructions", [])
        reconstructed_query_ids = {recon.get("subquery_id", "") for recon in reconstructions if recon.get("subquery_id")}  # 🔧 안전한 키 접근

    logger.info(f"[_merge_evaluation_results] Merging results: "
               f"existing={len(existing_results)}, new={len(new_results)}, "
               f"reconstructed={len(reconstructed_query_ids)}")

    # 병합 로직
    for query_id, new_result in new_results.items():
        existing_result = merged_results.get(query_id)

        if query_id in reconstructed_query_ids:
            # 재구성된 쿼리: 항상 새로운 결과로 업데이트
            merged_results[query_id] = new_result
            logger.debug(f"[_merge_evaluation_results] Updated reconstructed query: {query_id}")

        elif not existing_result:
            # 기존 결과가 없는 새로운 쿼리: 추가
            merged_results[query_id] = new_result
            logger.debug(f"[_merge_evaluation_results] Added new query: {query_id}")

        else:
            # 기존 결과가 있는 경우: 성공 보존 원칙 적용
            existing_relevant = existing_result.get("is_relevant", False)
            new_relevant = new_result.get("is_relevant", False)

            if existing_relevant and not new_relevant:
                # 기존 성공 → 새로 실패: 기존 성공 결과 보존
                logger.debug(f"[_merge_evaluation_results] Preserved successful result: {query_id}")
                # existing_result를 그대로 유지 (업데이트 안 함)

            elif not existing_relevant and new_relevant:
                # 기존 실패 → 새로 성공: 새로운 성공 결과로 업데이트
                merged_results[query_id] = new_result
                logger.debug(f"[_merge_evaluation_results] Updated to successful result: {query_id}")

            elif existing_relevant and new_relevant:
                # 둘 다 성공: 더 높은 점수의 결과 사용
                existing_score = existing_result.get("relevance_score", 0.0)
                new_score = new_result.get("relevance_score", 0.0)

                if new_score > existing_score:
                    merged_results[query_id] = new_result
                    logger.debug(f"[_merge_evaluation_results] Updated to higher score: {query_id} "
                               f"({existing_score:.3f} → {new_score:.3f})")
                else:
                    logger.debug(f"[_merge_evaluation_results] Kept existing higher score: {query_id}")

            else:
                # 둘 다 실패: 더 높은 신뢰도의 결과 사용
                existing_confidence = existing_result.get("confidence", 0.0)
                new_confidence = new_result.get("confidence", 0.0)

                if new_confidence > existing_confidence:
                    merged_results[query_id] = new_result
                    logger.debug(f"[_merge_evaluation_results] Updated to higher confidence: {query_id}")
                else:
                    logger.debug(f"[_merge_evaluation_results] Kept existing result: {query_id}")

    logger.info(f"[_merge_evaluation_results] Merge complete: {len(merged_results)} total results")
    return merged_results

def _determine_retry_necessity(
    non_relevant_queries: List[Dict[str, Any]],
    total_queries: int,
    evaluation_stats: Dict[str, Any]  # 향후 사용을 위해 유지
) -> Dict[str, Any]:
    """재시도 필요성 판단 로직

    Args:
        non_relevant_queries: 관련성이 낮은 쿼리들
        total_queries: 전체 쿼리 수
        evaluation_stats: 평가 통계

    Returns:
        Dict[str, Any]: 재시도 결정 정보
    """
    if not non_relevant_queries:
        return {
            "should_retry": False,
            "reason": "All queries are relevant",
            "strategy": None
        }

    non_relevant_count = len(non_relevant_queries)
    non_relevant_ratio = non_relevant_count / total_queries

    # 재시도 결정 기준들 (중앙 집중식 설정에서 로드)
    RETRY_THRESHOLD_RATIO = 1.0 - _qe_cfg.relevance_threshold  # 비관련 비율 임계값
    RETRY_MIN_QUERIES = 2  # 최소 비관련 쿼리 수 (고정값)
    LOW_CONFIDENCE_THRESHOLD = _qe_cfg.confidence_threshold  # 낮은 신뢰도 기준

    # 기준 1: 비관련 쿼리 비율이 높을 때
    if non_relevant_ratio >= RETRY_THRESHOLD_RATIO and non_relevant_count >= RETRY_MIN_QUERIES:
        return {
            "should_retry": True,
            "reason": f"High non-relevant ratio: {non_relevant_ratio:.1%} ({non_relevant_count}/{total_queries})",
            "strategy": "regenerate_all_non_relevant"
        }

    # 기준 2: 비관련 쿼리들의 신뢰도가 낮을 때 (평가가 불확실)
    low_confidence_queries = [
        q for q in non_relevant_queries
        if q.get("confidence", 1.0) < LOW_CONFIDENCE_THRESHOLD
    ]

    if len(low_confidence_queries) >= 2:
        return {
            "should_retry": True,
            "reason": f"Low confidence in {len(low_confidence_queries)} non-relevant evaluations",
            "strategy": "regenerate_low_confidence"
        }

    # 기준 3: 핵심 앵커 쿼리(Q1)가 비관련성일 때
    anchor_non_relevant = any(
        q.get("query_id", "").startswith("Q1") for q in non_relevant_queries
    )

    if anchor_non_relevant:
        return {
            "should_retry": True,
            "reason": "Anchor query (Q1) is non-relevant - critical for search quality",
            "strategy": "regenerate_anchor_focused"
        }

    # 재시도 불필요
    return {
        "should_retry": False,
        "reason": f"Acceptable non-relevant ratio: {non_relevant_ratio:.1%} ({non_relevant_count}/{total_queries})",
        "strategy": None
    }

def get_retry_target_queries(state: AgentState) -> List[Dict[str, Any]]:
    """재시도 대상 쿼리 목록 반환

    Args:
        state: retry_routing 정보가 포함된 agent state

    Returns:
        List[Dict[str, Any]]: 재시도할 쿼리들의 정보
    """
    metadata = state.get("metadata", {})
    retry_data = metadata.get("retry_routing", {})

    return retry_data.get("retry_queries", [])

def clear_retry_metadata(state: AgentState) -> AgentState:
    """재시도 관련 메타데이터 정리

    Args:
        state: 정리할 agent state

    Returns:
        AgentState: 재시도 메타데이터가 제거된 state
    """
    metadata = state.get("metadata", {})
    new_metadata = metadata.copy()

    # retry 관련 메타데이터 제거
    if "retry_routing" in new_metadata:
        del new_metadata["retry_routing"]

    return type(state)(
        messages=state.get("messages", []),
        context_messages=state.get("context_messages", []),
        metadata=new_metadata,
        current_agent=state.get("current_agent", ""),
        session_id=state.get("session_id", "")
    )

# ============================================
# 우아한 종료 및 부분 결과 처리
# ============================================

def get_loop_prevention_summary(state: AgentState) -> Dict[str, Any]:
    """루프 방지 시스템 상태 요약 반환

    Args:
        state: AgentState

    Returns:
        Dict[str, Any]: 루프 방지 상태 요약
    """
    metadata = state.get("metadata", {})
    loop_data = metadata.get("loop_prevention", {})
    last_decision = metadata.get("last_routing_decision", {})

    return {
        "global_iterations": loop_data.get("global_iteration_count", 0),
        "routing_history": loop_data.get("routing_history", [])[-10:],  # 최근 10개
        "forced_terminations": loop_data.get("forced_terminations", 0),
        "last_route": last_decision.get("route", "unknown"),
        "last_reason": last_decision.get("reason", ""),
        "termination_risk": "high" if loop_data.get("global_iteration_count", 0) >= 20 else "low"
    }

def create_graceful_termination_message(
    state: AgentState,
    termination_reason: str,
    available_data: Dict[str, Any]
) -> str:
    """우아한 종료 시 사용자에게 제공할 메시지 생성

    Args:
        state: AgentState
        termination_reason: 종료 이유
        available_data: 사용 가능한 부분 결과 데이터

    Returns:
        str: 종료 메시지
    """
    loop_summary = get_loop_prevention_summary(state)

    # 평가 결과 요약
    metadata = state.get("metadata", {})
    evaluator_data = metadata.get("quality_evaluator", {})
    cumulative_results = evaluator_data.get("cumulative_evaluation_results", {})

    successful_count = sum(
        1 for result in cumulative_results.values()
        if result.get("is_relevant", False)
    )
    total_count = len(cumulative_results)

    message_parts = [
        "🔄 **Processing Terminated** (Loop Prevention)",
        f"**Reason**: {termination_reason}",
        f"**Iterations**: {loop_summary['global_iterations']}/25",
        "",
        "📊 **Available Results**:",
        f"- Successful queries: {successful_count}/{total_count}",
        f"- Success rate: {successful_count/total_count*100:.1f}%" if total_count > 0 else "- No queries processed",
        "",
    ]

    # 성공한 쿼리가 있으면 부분 결과 제공
    if successful_count > 0:
        message_parts.extend([
            "✅ **Providing partial results based on successful queries**",
            "Note: Answer quality may be limited due to incomplete processing.",
            ""
        ])
    else:
        message_parts.extend([
            "⚠️ **No successful queries found**",
            "Unable to provide reliable results. Please try reformulating your question.",
            ""
        ])

    # 디버깅 정보 (선택적)
    if loop_summary["routing_history"]:
        message_parts.extend([
            f"🔍 **Debug Info**: Recent routes: {' → '.join(loop_summary['routing_history'])}",
            ""
        ])

    return "\n".join(message_parts)

def has_sufficient_partial_results(state: AgentState, minimum_success_rate: float = 0.3) -> bool:
    """부분 결과가 답변 생성에 충분한지 확인

    Args:
        state: AgentState
        minimum_success_rate: 최소 성공률 (기본값: 30%)

    Returns:
        bool: 부분 결과로 답변 생성 가능 여부
    """
    metadata = state.get("metadata", {})
    evaluator_data = metadata.get("quality_evaluator", {})
    cumulative_results = evaluator_data.get("cumulative_evaluation_results", {})

    if not cumulative_results:
        return False

    successful_count = sum(
        1 for result in cumulative_results.values()
        if result.get("is_relevant", False)
    )
    total_count = len(cumulative_results)

    success_rate = successful_count / total_count if total_count > 0 else 0
    return success_rate >= minimum_success_rate and successful_count >= 1


def _filter_duplicate_queries(
    evaluation_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Filter duplicate queries based on 95%+ similarity, keeping highest relevance score.

    Strategy:
    1. Compare all query pairs using SequenceMatcher (95%+ similarity threshold)
    2. Among duplicates, keep query with HIGHER relevance_score
    3. If relevance_score is equal, keep first query (arbitrary but deterministic)
    4. Mark lower-scored duplicates as is_relevant=False with failure reason

    Args:
        evaluation_results: Dict[query_id, result_dict] containing evaluation data

    Returns:
        Dict[query_id, result_dict] with duplicates filtered
    """
    from difflib import SequenceMatcher
    from ...logger.logger import get_logger

    logger = get_logger(__name__)

    if not evaluation_results:
        logger.info("[_filter_duplicate_queries] No results to filter")
        return evaluation_results

    # Step 1: Extract query texts for comparison
    query_texts = {}
    for qid, result in evaluation_results.items():
        query_text = result.get("query_text", "")
        query_texts[qid] = query_text

    # Step 2: Find all duplicate pairs (95%+ similarity)
    duplicate_groups = []
    processed_qids = set()
    qids = sorted(query_texts.keys())  # Deterministic ordering

    for i, qid1 in enumerate(qids):
        if qid1 in processed_qids:
            continue

        group = [qid1]
        text1 = query_texts[qid1]

        for qid2 in qids[i+1:]:
            if qid2 in processed_qids:
                continue

            text2 = query_texts[qid2]

            # Calculate similarity
            similarity = SequenceMatcher(None, text1, text2).ratio()

            if similarity >= 0.95:
                group.append(qid2)
                processed_qids.add(qid2)

        if len(group) > 1:
            duplicate_groups.append(group)
            for qid in group:
                processed_qids.add(qid)

    # Step 3: Process each duplicate group
    total_duplicates_found = 0
    queries_marked_failed = 0

    for group in duplicate_groups:
        logger.info(f"[_filter_duplicate_queries] Found duplicate group: {group}")

        # Extract relevance scores
        scores = {}
        for qid in group:
            result = evaluation_results[qid]
            scores[qid] = result.get("relevance_score", 0.0)

        # Find query with highest relevance score
        max_score = max(scores.values())
        winners = [qid for qid, score in scores.items() if score == max_score]

        # If multiple queries have same max score, keep first (deterministic)
        winner_qid = winners[0]
        losers = [qid for qid in group if qid != winner_qid]

        logger.info(f"[_filter_duplicate_queries] Winner: {winner_qid} (score={max_score:.3f})")
        logger.info(f"[_filter_duplicate_queries] Marking as failed: {losers}")

        # Mark losers as failed
        for loser_qid in losers:
            result = evaluation_results[loser_qid]

            result["is_relevant"] = False
            result["failure_reason"] = (
                f"Duplicate query filtered: similar to {winner_qid} "
                f"with higher relevance score ({max_score:.3f} vs {scores[loser_qid]:.3f})"
            )

            queries_marked_failed += 1

        total_duplicates_found += len(group) - 1

    # Step 4: Logging summary
    logger.info(f"[_filter_duplicate_queries] Filtering complete:")
    logger.info(f"  - Duplicate groups found: {len(duplicate_groups)}")
    logger.info(f"  - Total duplicates detected: {total_duplicates_found}")
    logger.info(f"  - Queries marked as failed: {queries_marked_failed}")
    logger.info(f"  - Queries retained: {len(evaluation_results) - queries_marked_failed}")

    return evaluation_results

# ─────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────

__all__ = [
    "quality_evaluator_node",
    "should_retry_routing",
    "get_evaluation_results_for_query",
    "get_relevance_summary",
    "format_evaluation_results",
    "get_retry_target_queries",
    "clear_retry_metadata",
    # Loop prevention functions
    "get_loop_prevention_summary",
    "create_graceful_termination_message",
    "has_sufficient_partial_results"
]