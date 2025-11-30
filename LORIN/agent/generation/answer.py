from __future__ import annotations
import asyncio
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ...llm import Chatbot
from ...logger.logger import get_logger
from ..state import AgentState
from ..schema import (
    MetadataManager,
    LoopPreventionStatus,
    create_answer_metadata,
    get_search_results,
    get_quality_filtered_search_results
)
from ...prompt.template_loader import PromptTemplateLoader

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def _after_thought(output: str) -> str:
    """ 응답 문자열에서 마지막 </thought> 이후 부분만 반환.
    - 태그가 없으면 원문 strip 반환
    - 대소문자/개행/공백 허용: </thought>, </Thought > 등
    """
    if not output:
        return ""
    m = list(re.finditer(r"(?is)</\s*thought\s*>", output))
    if not m:
        return output.strip()
    end = m[-1].end()
    return output[end:].strip()


def _get_last_user_utterance(messages: List[BaseMessage]) -> Optional[str]:
    """ 대화 메시지 리스트에서 마지막 사용자 발화를 찾아 반환
    (HumanMessage, type='human', role='user' 순으로 탐색)
    """
    if not messages:
        return None
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human" or getattr(m, "role", "") == "user":
            content = getattr(m, "content", None)
            if content:
                return str(content)
    return str(getattr(messages[-1], "content", "") or "")


def _kv(meta: Dict[str, Any], *keys: str, default=None):
    """메타 사전에서 대소문자 무시 키 조회."""
    if not meta:
        return default
    lowers = {k.lower(): k for k in meta.keys()}
    for k in keys:
        lk = k.lower()
        if lk in lowers:
            return meta[lowers[lk]]
    return default


def _basename(p: str) -> str:
    try:
        import os as _os
        return _os.path.basename(p) or p
    except Exception:
        return str(p)


def _make_label_or_none(meta: Dict[str, Any]) -> Optional[str]:
    """ ⚠️ doc1/doc2 같은 가짜 라벨은 생성하지 않음.
    라벨 후보가 없으면 None을 반환하여 출력에서 라벨을 생략한다.
    우선순위: url → path/file → source → title
    """
    url = _kv(meta, "url", "URI", "href")
    if url:
        return str(url)
    path = _kv(meta, "path", "filepath", "file_path", "fullpath")
    file_ = _kv(meta, "file", "filename", "name")
    if path:
        return _basename(str(path))
    if file_:
        return _basename(str(file_))
    source = _kv(meta, "source")
    if source:
        return str(source)
    title = _kv(meta, "title")
    if title:
        return str(title)
    return None  # ⬅️ 가짜 라벨 금지


def _parse_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


# ============================================
# 루프 방지 처리 함수들
# ============================================

def _has_partial_evaluation_results(state: AgentState) -> bool:
    """부분 평가 결과가 있는지 확인

    Args:
        state: AgentState 객체

    Returns:
        bool: 부분적으로 성공한 쿼리가 있으면 True
    """
    evaluator_data = MetadataManager.safe_get_metadata(state, "quality_evaluator")
    if not evaluator_data:
        return False

    cumulative_results = evaluator_data.get("cumulative_evaluation_results", {})
    if not cumulative_results:
        return False

    successful_count = sum(
        1 for result in cumulative_results.values()
        if isinstance(result, dict) and result.get("is_relevant", False)
    )

    return successful_count > 0


def _check_loop_prevention_status(state: AgentState) -> LoopPreventionStatus:
    """루프 방지 상태 확인

    MetadataManager를 사용하여 안전하게 루프 방지 관련 메타데이터를 조회합니다.

    Args:
        state: AgentState 객체

    Returns:
        LoopPreventionStatus: 루프 방지 상태 정보 (TypedDict)
    """
    loop_data = MetadataManager.safe_get_metadata(state, "loop_prevention") or {}
    last_decision = MetadataManager.safe_get_metadata(state, "last_routing_decision") or {}

    # 강제 종료 조건들 확인
    global_iterations = loop_data.get("global_iteration_count", 0)
    forced_terminations = loop_data.get("forced_terminations", 0)

    # 종료 여부 판단
    terminated = False
    reason = ""

    if global_iterations >= 25:
        terminated = True
        reason = "Maximum global iterations (25) reached"
    elif forced_terminations > 0:
        terminated = True
        reason = last_decision.get("reason", "Loop prevention triggered")
    elif last_decision.get("strategy") in [
        "global_limit_prevention",
        "oscillation_prevention",
        "conservative_termination"
    ]:
        terminated = True
        reason = last_decision.get("reason", "Loop prevention strategy activated")

    return LoopPreventionStatus(
        terminated=terminated,
        reason=reason,
        global_iterations=global_iterations,
        forced_terminations=forced_terminations,
        routing_history=loop_data.get("routing_history", [])[-5:],
        strategy=last_decision.get("strategy", ""),
        has_partial_results=_has_partial_evaluation_results(state)
    )

def _create_loop_termination_message(
    state: AgentState,
    reason: str,
    loop_status: Dict[str, Any]
) -> str:
    """루프 종료 시 사용자 메시지 생성"""
    from .quality_evaluator_node import create_graceful_termination_message

    return create_graceful_termination_message(state, reason, loop_status)

def _build_termination_response(
    termination_message: str,
    qids: List[str],
    subanswers_by_qid: Dict[str, Dict[str, Any]],
    loop_status: Dict[str, Any]
) -> str:
    """종료 상황에서의 최종 응답 구성"""
    response_parts = [termination_message]

    # 부분 결과가 있으면 포함
    if loop_status.get("has_partial_results", False) and subanswers_by_qid:
        response_parts.append("\n## 📝 **Available Partial Results**\n")

        # 서브쿼리 응답이 있으면 간략하게 포함
        for qid in qids:
            subans = subanswers_by_qid.get(qid)
            if subans and subans.get("output"):
                query_text = subans.get("query", f"Query {qid}")
                output = subans.get("output", "")

                # 긴 응답은 요약
                if len(output) > 300:
                    output = output[:300] + "..."

                response_parts.append(f"**{query_text}**: {output}\n")

    return "\n".join(response_parts)


def _extract_line_range(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """ 메타에서 라인 범위를 추출.
    - 오직 "Range"/"range" 키만 지원
    - 리스트/튜플: [start, end]
    - 문자열: "[123,456]", "123-456", "123~456", "123 456"
    - 실패 시 (None, None)
    """
    rng = _kv(meta, "Range", "range")
    if rng is None:
        return None, None
    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
        ls = _parse_int(rng[0]); le = _parse_int(rng[1])
        return ls, le
    if isinstance(rng, str):
        s = rng.strip()
        s = re.sub(r"^[\[\(\{]\s*|\s*[\]\)\}]$", "", s)  # 양 끝 괄호 제거
        m = re.match(r"\s*(\d+)\s*[,~\-]\s*(\d+)\s*$", s)  # 123-456, 123~456, 123,456
        if m:
            return _parse_int(m.group(1)), _parse_int(m.group(2))
        m = re.match(r"\s*(\d+)\s+(\d+)\s*$", s)  # 123 456
        if m:
            return _parse_int(m.group(1)), _parse_int(m.group(2))
    return None, None


def _format_meta_all(meta: Dict[str, Any], extra: Dict[str, Any] | None = None) -> str:
    """디버깅/추적용 메타를 사람이 읽기 쉬운 한 줄 문자열로."""
    merged = dict(meta or {})
    if extra:
        for k, v in extra.items():
            if v is not None and k not in merged:
                merged[k] = v

    priority = [
        "url", "path", "file", "source", "title", "Range", "range",
        "line_start", "line_end", "score", "rerank_score"
    ]

    def sort_key(item):
        k = item[0]
        try:
            idx = priority.index(k)
        except ValueError:
            idx = 999
        return (idx, k)

    items = sorted(merged.items(), key=sort_key)
    return " | ".join(f"{k}={v}" for k, v in items)


# ──────────────────────────────────────────────
# 증거 수집
# ──────────────────────────────────────────────

def _qid_sort_key(qid: str) -> Tuple[int, str]:
    """QID 정렬 키. 'Q1','Q2' 숫자를 우선 정렬, 그 외는 사전순."""
    m = re.match(r"[Qq](\d+)", qid or "")
    if m:
        return (int(m.group(1)), qid)
    return (10**9, qid or "")


def _collect_evidence(state: AgentState, results_by_qid: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """모든 QID 결과를 한데 모아 evidence_block 구성 (품질 필터링 적용).

    Args:
        state: AgentState 객체
        results_by_qid: QID별 검색 결과

    Returns:
        Tuple[str, Dict[str, Any]]: (evidence_block 문자열, label_map)
    """
    logger.info(f"[_collect_evidence] Using quality-filtered results: {len(results_by_qid)} QIDs")

    # MetadataManager를 사용하여 쿼리 매핑 조회
    retr_data = MetadataManager.safe_get_metadata(state, "faiss_retriever")
    queries_by_qid: Dict[str, str] = retr_data.get("queries_by_qid", {}) if retr_data else {}
    logger.info(f"[_collect_evidence] queries_by_qid: {len(queries_by_qid)} queries")
    lines: List[str] = []
    label_map: Dict[str, Any] = {}
    anon_counter = 0

    qids = sorted(results_by_qid.keys(), key=_qid_sort_key)
    logger.info(f"[_collect_evidence] Processing {len(qids)} QIDs")

    for qid in qids:
        used_query = (queries_by_qid or {}).get(qid, "")
        docs = results_by_qid.get(qid) or []
        logger.info(f"[_collect_evidence] QID {qid}: {len(docs)} docs")

        if not docs:
            continue

        # ✨ Sort documents by line_start for chronological interpretation
        def get_line_start(doc):
            meta = doc.get("metadata") or {}
            ls, _ = _extract_line_range(meta)
            return ls if ls is not None else float('inf')

        sorted_docs = sorted(docs, key=get_line_start)
        logger.info(f"[_collect_evidence] QID {qid}: Sorted {len(sorted_docs)} docs by line position")

        for i, d in enumerate(sorted_docs):
            logger.info(f"[_collect_evidence] QID {qid}, doc {i}: type={type(d)}")
            try:
                meta = d.get("metadata") or {}
                logger.info(f"[_collect_evidence] QID {qid}, doc {i}: metadata success")
            except Exception as e:
                logger.error(f"[_collect_evidence] QID {qid}, doc {i}: metadata error: {e}")
                raise
            score = d.get("score")
            rscore = d.get("rerank_score")
            ls, le = _extract_line_range(meta)
            label = _make_label_or_none(meta)
            if label is None:
                anon_counter += 1
                key = f"anon_{anon_counter}"
            else:
                key = f"label_{hash(label)}"

            header_parts = []
            header_parts.append(f"### {label}" if label else "### (소스 라벨 없음)")
            if ls is not None or le is not None:
                header_parts.append(f"(lines {ls if ls is not None else '?'}~{le if le is not None else '?'})")
            if used_query:
                header_parts.append(f"input_query: {used_query}")
            lines.append(" ".join(header_parts))

            meta_str = _format_meta_all(meta, {"score": score, "rerank_score": rscore})
            if meta_str:
                lines.append(f"META: {meta_str}")

            # 🔧 FIX: SearchResult uses "content" field, not "text"
            txt = str(d.get("content", "") or "").strip()
            if txt:
                lines.append(txt)
            lines.append("")

            label_map[key] = {
                "label": label,
                "meta": meta,
                "score": score,
                "rerank_score": rscore,
                "line_start": ls,
                "line_end": le,
                "input_query": used_query or None,
                "input_querry": used_query or None,
            }

    context = "\n".join(lines).strip()
    return context, label_map


def _collect_evidence_for_qid(state: AgentState, qid: str, results_by_qid: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """특정 QID의 결과만 evidence_block으로 구성 (품질 필터링 적용).

    Args:
        state: AgentState 객체
        qid: 대상 쿼리 ID
        results_by_qid: QID별 검색 결과

    Returns:
        Tuple[str, Dict[str, Any]]: (evidence_block 문자열, label_map)
    """
    # MetadataManager를 사용하여 쿼리 매핑 조회
    retr_data = MetadataManager.safe_get_metadata(state, "faiss_retriever")
    queries_by_qid: Dict[str, str] = retr_data.get("queries_by_qid", {}) if retr_data else {}

    docs = results_by_qid.get(qid) or []
    used_query = (queries_by_qid or {}).get(qid, "")
    lines: List[str] = []
    label_map: Dict[str, Any] = {}
    anon_counter = 0

    if not docs:
        return "", {}

    # ✨ Sort documents by line_start for chronological interpretation
    def get_line_start(doc):
        meta = doc.get("metadata") or {}
        ls, _ = _extract_line_range(meta)
        return ls if ls is not None else float('inf')

    sorted_docs = sorted(docs, key=get_line_start)

    for d in sorted_docs:
        meta = d.get("metadata") or {}
        score = d.get("score")
        rscore = d.get("rerank_score")
        ls, le = _extract_line_range(meta)
        label = _make_label_or_none(meta)
        if label is None:
            anon_counter += 1
            key = f"anon_{anon_counter}"
        else:
            key = f"label_{hash(label)}"

        header_parts = []
        header_parts.append(f"### {label}" if label else "### (소스 라벨 없음)")
        if ls is not None or le is not None:
            header_parts.append(f"(lines {ls if ls is not None else '?'}~{le if le is not None else '?'})")
        if used_query:
            header_parts.append(f"input_query: {used_query}")
        lines.append(" ".join(header_parts))

        meta_str = _format_meta_all(meta, {"score": score, "rerank_score": rscore})
        if meta_str:
            lines.append(f"META: {meta_str}")

        # 🔧 FIX: SearchResult uses "content" field, not "text"
        txt = str(d.get("content", "") or "").strip()
        if txt:
            lines.append(txt)
        lines.append("")

        label_map[key] = {
            "label": label,
            "meta": meta,
            "score": score,
            "rerank_score": rscore,
            "line_start": ls,
            "line_end": le,
            "input_query": used_query or None,
            "input_querry": used_query or None,
        }

    context = "\n".join(lines).strip()
    return context, label_map


# ──────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────

def _build_sub_prompt(user_or_subquery: str, evidence_block: str, intent: str = "debug", query_type: str = "cause") -> str:
    """ 서브쿼리용 프롬프트 (Jinja2 template 사용 with intent-based template selection).
    - evidence를 참고해 질문(또는 서브쿼리)에 대한 간결한 답변을 작성.
    - 추가 정보 요청 금지(항상 답변 시도).

    Args:
        user_or_subquery: User question or subquery text
        evidence_block: Retrieved evidence
        intent: Query intent - "debug" or "analysis" (default: "debug")
        query_type: Query type - "cause" or "effect" (default: "cause")
    """
    template_loader = PromptTemplateLoader()

    # Select template based on intent
    if intent == "analysis":
        template_name = "answer_activity.j2"
    else:  # debug (default)
        template_name = "answer_sub.j2"

    return template_loader.render_template(
        template_name,
        query=user_or_subquery.strip(),
        evidence_block=evidence_block if evidence_block else "(No retriever evidence)",
        query_type=query_type  # ✨ Pass query type for perspective instruction
    )


def _build_final_prompt(user_question: str, ordered_qids: List[str], subanswers_by_qid: Dict[str, Dict[str, Any]], available_ranges_by_qid: Dict[str, List[str]] = None) -> str:
    """ 최종 프롬프트 (Jinja2 template 사용):
    ▶ 최종 입력 = (사용자 질문) + (각 서브쿼리별 '그대로의' 답변들) + (available ranges for verification) + (최종 지침 프롬프트)
    """
    template_loader = PromptTemplateLoader()

    return template_loader.render_template(
        "answer_final.j2",
        user_question=user_question.strip() or '(Empty question)',
        ordered_qids=ordered_qids,
        subanswers_by_qid=subanswers_by_qid,
        available_ranges_by_qid=available_ranges_by_qid or {}  # Token-efficient: only ranges, not full evidence
    )


# ──────────────────────────────────────────────
# LLM 호출 (3회 재시도)
# ──────────────────────────────────────────────

async def _ask_with_retries(llm: Chatbot, prompt: str, retries: int = 2) -> Tuple[str, str]:
    """ LLM 호출을 최대 retries회 재시도 (기본 2회로 최적화).
    반환: (final_text_after_thought, raw_text_from_llm)
    실패 시 폴백 메시지 반환.
    """
    last_err: Optional[Exception] = None
    raw_text: str = ""
    # 재시도 횟수 최적화: 기본 3→2
    optimized_retries = min(max(1, retries), 2)

    for i in range(1, optimized_retries + 1):
        try:
            res = await llm.ask(prompt)
            raw_text = getattr(res, "content", None) or str(res) or ""
            final_text = _after_thought(raw_text)
            if final_text.strip():
                return final_text, raw_text
            logger.warning("[answer_node] empty LLM content (attempt %d/%d)", i, optimized_retries)
        except Exception as e:
            last_err = e
            logger.warning("[answer_node] LLM call failed (attempt %d/%d): %s", i, optimized_retries, e)

            # Exponential backoff 적용
            if i < optimized_retries:
                await asyncio.sleep(2 ** (i - 1))

    if last_err:
        logger.warning("[answer_node] All retry attempts failed: %s", last_err)

    fallback = (
        "죄송합니다. 현재 답변을 생성하는 데 문제가 발생했습니다.\n"
        "근거: 없음"
    )
    return fallback, raw_text


# ──────────────────────────────────────────────
# 퍼블릭 엔트리: answer_node
# ──────────────────────────────────────────────

def answer_node(
    *,
    answer_llm: Optional[Chatbot] = None,
    agent_name: str = "answer",
) -> callable:
    if answer_llm is None:
        from ...utils import create_chatbot_from_env
        answer_llm = create_chatbot_from_env(
            temperature=0.1
            # max_tokens는 기본값(4096) 사용
        )

    async def node_function(state: AgentState) -> AgentState:
        logger.info(f"[answer_node] Called with state type: {type(state)}")

        # 🛡️ State 타입 검증 (LangGraph 상태 전달 체인 보호)
        if isinstance(state, str):
            logger.error(f"[answer_node] Received string state instead of AgentState: {state[:200]}...")
            raise TypeError(f"Expected AgentState but received string: {type(state)}")

        if not hasattr(state, 'get'):
            logger.error(f"[answer_node] State object missing 'get' method: {type(state)}")
            raise TypeError(f"Invalid state object type: {type(state)}")

        try:
            messages: List[BaseMessage] = state.get("messages", []) or []
            logger.info(f"[answer_node] Successfully got messages: {len(messages)}")
        except Exception as e:
            logger.error(f"[answer_node] Error getting messages: {e}")
            raise

        try:
            md: Dict[str, Any] = state.get("metadata", {}) or {}
            logger.info(f"[answer_node] Successfully got metadata: {type(md)}")
        except Exception as e:
            logger.error(f"[answer_node] Error getting metadata: {e}")
            raise

        try:
            user_utter = _get_last_user_utterance(messages) or ""
            logger.info(f"[answer_node] Successfully got user utterance")
        except Exception as e:
            logger.error(f"[answer_node] Error getting user utterance: {e}")
            raise

        # ============================================
        # 품질 필터링된 결과 한 번만 조회 (중복 호출 최적화)
        # ============================================
        try:
            logger.info(f"[answer_node] Getting quality-filtered search results from state")
            results_by_qid: Dict[str, List[Dict[str, Any]]] = get_quality_filtered_search_results(state)
            logger.info(f"[answer_node] Quality-filtered search results extraction successful: {len(results_by_qid)} queries")
        except Exception as e:
            logger.error(f"[answer_node] Error getting quality-filtered search results: {e}")
            raise

        # (A) 품질 필터링된 결과로 evidence 수집
        try:
            logger.info(f"[answer_node] Collecting evidence from filtered results")
            evidence_block_all, label_map_all = _collect_evidence(state, results_by_qid)
            logger.info(f"[answer_node] Evidence collection successful")
        except Exception as e:
            logger.error(f"[answer_node] Error collecting evidence: {e}")
            raise

        # ============================================
        # 루프 방지 상태 확인 (성공한 쿼리가 있는지 고려)
        # ============================================
        try:
            logger.info(f"[answer_node] Checking loop prevention status")
            loop_prevention_active = _check_loop_prevention_status(state)
            logger.info(f"[answer_node] Loop prevention check completed: {loop_prevention_active.get('terminated', False)}")
        except Exception as e:
            logger.error(f"[answer_node] Error in loop prevention check: {e}")
            raise

        # 스킵된 쿼리 정보 로깅
        skipped_queries = md.get("skipped_queries", [])
        if skipped_queries:
            logger.warning(f"[answer_node] Skipped queries (gave up after max failures): {skipped_queries}")

        # 성공한 쿼리가 있으면 정상 답변 생성, 없으면 종료 메시지 생성
        termination_message = None
        has_successful_queries = len(results_by_qid) > 0

        if loop_prevention_active["terminated"] and not has_successful_queries:
            # 루프 방지 종료 + 성공한 쿼리 없음 → 종료 메시지
            termination_message = _create_loop_termination_message(
                state,
                loop_prevention_active["reason"],
                loop_prevention_active
            )
            logger.warning(f"[answer_node] Loop prevention termination with no successful queries: {loop_prevention_active['reason']}")
        elif loop_prevention_active["terminated"] and has_successful_queries:
            # 루프 방지 종료 + 성공한 쿼리 있음 → 정상 답변 생성
            logger.info(f"[answer_node] Loop prevention active but proceeding with {len(results_by_qid)} successful queries")

        # 쿼리 매핑 추출 (호환성 지원)
        retr_data = MetadataManager.safe_get_metadata(state, "faiss_retriever")
        queries_by_qid: Dict[str, str] = retr_data.get("queries_by_qid", {}) if retr_data else {}

        # ✨ Extract intent from planner metadata
        planner_data = MetadataManager.safe_get_metadata(state, "planner")
        intent = planner_data.get("intent", "debug") if planner_data else "debug"
        logger.info(f"[answer_node] Using intent: '{intent}' for answer generation")

        # ✨ Extract query types from planner metadata (expansion queries)
        plan_json = planner_data.get("last_plan_json", {}) if planner_data else {}
        subqueries = plan_json.get("subqueries", [])
        query_type_map = {sq.get("id"): sq.get("type", "cause") for sq in subqueries}
        logger.info(f"[answer_node] Query type mapping: {query_type_map}")

        # 🔧 성공한 쿼리만 사용 (실패 쿼리 제외)
        qid_set = set(results_by_qid.keys())
        qids = sorted(list(qid_set), key=_qid_sort_key)

        # (C) 서브쿼리별 처리: evidence 수집 → 병렬 응답 생성 ⚡
        subanswers_by_qid: Dict[str, Dict[str, Any]] = {}
        evidence_by_qid: Dict[str, Any] = {}

        if qids:
            # 🚀 병렬화: 모든 서브쿼리 답변을 동시에 생성
            async def process_single_qid(qid: str):
                """단일 서브쿼리 처리 (병렬 실행용)"""
                used_query = (queries_by_qid or {}).get(qid, "") or f"(QID:{qid})"
                ev_block_qid, label_map_qid = _collect_evidence_for_qid(state, qid, results_by_qid)

                # 🔍 디버깅: Evidence 길이 확인
                logger.info(f"[answer_node] {qid} evidence length: {len(ev_block_qid)} chars, docs: {len(results_by_qid.get(qid, []))}")
                if len(ev_block_qid) < 100:
                    logger.warning(f"[answer_node] {qid} has very short evidence: {ev_block_qid[:200]}")

                # ✨ Get query type for this QID
                query_type = query_type_map.get(qid, "cause")
                logger.info(f"[answer_node] {qid} using query_type: '{query_type}'")

                sub_prompt = _build_sub_prompt(used_query, ev_block_qid, intent=intent, query_type=query_type)
                sub_final_text, sub_raw_text = await _ask_with_retries(answer_llm, sub_prompt, retries=2)
                return qid, {
                    "query": used_query,
                    "output": sub_final_text.strip(),
                    "raw": sub_raw_text,
                }, label_map_qid

            # 🔧 Provider 기반 동시성 제한 (EXAONE은 순차 처리)
            llm_provider = md.get("llm_provider", "").lower()

            if llm_provider == "exaone":
                # EXAONE: 순차 처리 (메모리 안전)
                max_concurrent = 1
                logger.info(f"[answer_node] Sequential processing for EXAONE (max_concurrent=1)")
            else:
                # API 모델: GPU 메모리 기반 병렬 처리
                max_concurrent = 5  # 기본 동시 실행 수

                # GPU 메모리 여유 확인 및 동적 조정
                import torch
                if torch.cuda.is_available():
                    free_memory_gb = torch.cuda.mem_get_info(0)[0] / 1e9
                    if free_memory_gb < 4.0:
                        logger.warning(f"[answer_node] Low GPU memory ({free_memory_gb:.1f}GB), reducing max_concurrent to 2")
                        max_concurrent = 2
                    elif free_memory_gb < 6.0:
                        logger.warning(f"[answer_node] Moderate GPU memory ({free_memory_gb:.1f}GB), limiting max_concurrent to 3")
                        max_concurrent = 3
                    logger.info(f"[answer_node] Parallel processing with max_concurrent={max_concurrent} (free_memory={free_memory_gb:.1f}GB)")
                else:
                    logger.info(f"[answer_node] Parallel processing with max_concurrent={max_concurrent} (no GPU)")

            # Chunk 기반 병렬 처리
            all_results = []
            for chunk_idx in range(0, len(qids), max_concurrent):
                chunk = qids[chunk_idx:chunk_idx + max_concurrent]
                logger.info(f"[answer_node] Processing chunk {chunk_idx//max_concurrent + 1}/{(len(qids) + max_concurrent - 1)//max_concurrent} ({len(chunk)} queries)")

                tasks = [process_single_qid(qid) for qid in chunk]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(chunk_results)

                # 청크 간 메모리 정리
                if torch.cuda.is_available() and chunk_idx + max_concurrent < len(qids):
                    torch.cuda.empty_cache()

            # 결과 수집
            for result in all_results:
                if isinstance(result, Exception):
                    logger.error(f"[answer_node] Subquery processing failed: {result}")
                    continue
                qid, subanswer, label_map = result
                subanswers_by_qid[qid] = subanswer
                evidence_by_qid[qid] = label_map
        else:
            # 서브쿼리가 없으면 전체 evidence로 단발 처리(폴백)
            # Fallback uses default query_type="cause"
            sub_prompt = _build_sub_prompt(user_utter, evidence_block_all, intent=intent, query_type="cause")
            sub_final_text, sub_raw_text = await _ask_with_retries(answer_llm, sub_prompt, retries=3)
            subanswers_by_qid["Q0"] = {
                "query": user_utter,
                "output": sub_final_text.strip(),
                "raw": sub_raw_text,
            }
            evidence_by_qid["Q0"] = label_map_all
            qids = ["Q0"]

        # (D) 최종 결합: 사용자 질문 + 서브답변(그대로) + 최종 지침 프롬프트
        # ✨ Extract available ranges from evidence (token-efficient)
        available_ranges_by_qid = {}
        for qid, label_map in evidence_by_qid.items():
            ranges = []
            for key, data in label_map.items():
                if isinstance(data, dict) and "meta" in data:
                    line_range = data["meta"].get("line_range", "")
                    if line_range:
                        ranges.append(line_range)
            available_ranges_by_qid[qid] = ranges

        if termination_message:
            # 루프 방지 종료 시 특별 메시지 + 부분 결과 처리
            final_text = _build_termination_response(
                termination_message,
                qids,
                subanswers_by_qid,
                loop_prevention_active
            )
            final_raw_text = final_text
            final_prompt = "Loop prevention termination - no LLM call needed"
        else:
            # 정상 처리 - Pass available ranges for verification
            # Build final prompt with all subquery document summaries
            final_prompt = _build_final_prompt(user_utter, qids, subanswers_by_qid, available_ranges_by_qid)

            # Call LLM to synthesize final answer to user's question
            final_text, final_raw_text = await _ask_with_retries(answer_llm, final_prompt, retries=3)

        # (E) 추적 메타 보존 - MetadataManager를 사용한 안전한 저장
        existing_answer_data = MetadataManager.safe_get_metadata(state, "answer") or {}

        # 새로운 answer 메타데이터 구성
        answer_metadata = {
            **existing_answer_data,  # 기존 데이터 보존
            # 최적화: 전체 evidence는 faiss_retriever에서 참조
            "evidence_ref": "metadata.faiss_retriever.results_by_qid",
            "used_queries_ref": "metadata.faiss_retriever.queries_by_qid",
            # QID별 서브 응답 (실제 데이터)
            "subanswers_by_qid": subanswers_by_qid,
            # 최종 응답 (압축)
            "final": {
                "output": final_text.strip(),
                # 디버깅 시에만 저장
                "_debug": {
                    "prompt_length": len(final_prompt),
                    "raw_length": len(final_raw_text)
                } if os.getenv("DEBUG_MODE") == "1" else {}
            },
        }

        # MetadataManager를 통해 안전하게 저장
        MetadataManager.safe_set_metadata(state, "answer", answer_metadata)

        # (F) 실험용 출력 파싱 - 최종 답변에서 line 범위 추출 (다중 범위 지원)
        experiment_output = None

        # 📊 [NEW] 전체 로그 범위 계산 (모든 검색된 문서에서 추출)
        total_log_start = float('inf')
        total_log_end = 0
        overall_total_lines = 0
        all_doc_lines = []

        for qid, label_map in evidence_by_qid.items():
            for key, data in label_map.items():
                if isinstance(data, dict):
                    meta = data.get("meta") or {}
                    total_lines_meta = _parse_int((meta or {}).get("total_lines"))
                    if total_lines_meta:
                        overall_total_lines = max(overall_total_lines, total_lines_meta)

                    line_start = data.get("line_start")
                    line_end = data.get("line_end")
                    if line_start is not None and line_end is not None:
                        total_log_start = min(total_log_start, line_start)
                        total_log_end = max(total_log_end, line_end)
                        all_doc_lines.extend(range(line_start, line_end + 1))

        # 전체 로그 범위가 없으면 기본값 설정
        if overall_total_lines > 0:
            total_log_start = 1
            total_log_end = overall_total_lines
        elif total_log_start == float('inf'):
            total_log_start = 1
            total_log_end = 1

        # 중복 제거하여 실제 검색된 로그 라인 수 계산
        unique_doc_lines = sorted(list(set(all_doc_lines))) if all_doc_lines else []
        total_line_span = max(total_log_end - total_log_start + 1, 1)

        if final_text:
            # 다중 line 범위 패턴 파싱
            line_patterns = [
                r'line\[(\d+)~(\d+)\]',              # "line[100~200]" (권장 형식)
                r'lines?\s+(\d+)\s*[-~]\s*(\d+)',    # "line 100-200", "lines 100~200"
                r'\[(\d+)\s*[-~]\s*(\d+)\]',          # "[100-200]", "[100~200]"
                r'(\d+)\s*[-~]\s*(\d+)\s+lines?',    # "100-200 lines"
            ]

            # 모든 매칭 범위를 수집
            all_ranges = []
            for pattern in line_patterns:
                matches = re.finditer(pattern, final_text, re.IGNORECASE)
                for match in matches:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    if start <= end:  # 유효한 범위만
                        all_ranges.append((start, end))
                        logger.info(f"[answer_node] Extracted line range: {start}-{end}")

            if all_ranges:
                # 중복 제거 및 정렬
                all_ranges = sorted(list(set(all_ranges)))

                # 모든 범위의 라인을 하나의 리스트로 병합
                recommended_lines = []
                for start, end in all_ranges:
                    recommended_lines.extend(range(start, end + 1))

                # 중복 제거 및 정렬
                recommended_lines = sorted(list(set(recommended_lines)))

                # 전체 범위 계산 (첫 번째 라인 ~ 마지막 라인)
                overall_start = recommended_lines[0]
                overall_end = recommended_lines[-1]

                experiment_output = {
                    # 기존 필드 유지 (하위 호환성)
                    "recommended_lines": recommended_lines,
                    "start_line": overall_start,
                    "end_line": overall_end,
                    "line_count": len(recommended_lines),
                    "ranges": all_ranges,  # 개별 범위 정보 보존
                    # 🔬 실험용 메트릭
                    "faiss_calls": md.get("faiss_calls", 0),
                    "planner_iterations": md.get("planner_iterations", 0),
                    # 📊 [NEW] 전체 로그 범위 정보 추가
                    "total_log_range": {
                        "start": total_log_start,
                        "end": total_log_end,
                        "count": total_line_span,
                        "retrieved_lines": len(unique_doc_lines),
                        "retrieved_coverage": f"{len(unique_doc_lines) / total_line_span * 100:.1f}%" if total_line_span > 0 else "0%"
                    },
                    # 📊 [NEW] 추천 범위 상세 정보 추가
                    "recommended_range": {
                        "start": overall_start,
                        "end": overall_end,
                        "count": len(recommended_lines),
                        "coverage": f"{len(recommended_lines) / total_line_span * 100:.1f}%" if total_line_span > 0 else "0%"
                    }
                }
                logger.info(f"[answer_node] Total {len(all_ranges)} ranges found, {len(recommended_lines)} total lines")
                logger.info(f"[answer_node] Total log range: {total_log_start}-{total_log_end} ({total_line_span} lines), Coverage: {experiment_output['recommended_range']['coverage']}")
            else:
                # 📊 [NEW] 추천 범위가 없어도 전체 로그 범위는 제공
                experiment_output = {
                    # 🔬 실험용 메트릭
                    "faiss_calls": md.get("faiss_calls", 0),
                    "planner_iterations": md.get("planner_iterations", 0),
                    # 📊 [NEW] 전체 로그 범위 정보
                    "total_log_range": {
                        "start": total_log_start,
                        "end": total_log_end,
                        "count": total_line_span,
                        "retrieved_lines": len(unique_doc_lines),
                        "retrieved_coverage": f"{len(unique_doc_lines) / total_line_span * 100:.1f}%" if total_line_span > 0 else "0%"
                    },
                    "recommended_range": None  # 추천 범위 없음
                }
                logger.warning(f"[answer_node] No line range found in final answer, but total range available: {total_log_start}-{total_log_end}")

        # (G) 메시지 누적
        ai_msg = AIMessage(
            content=final_text.strip(),
            additional_kwargs={"agent_name": agent_name},
        )
        new_messages = (messages or []) + [ai_msg]

        return type(state)(
            messages=new_messages,
            context_messages=state.get("context_messages", []),
            metadata=md,
            current_agent=agent_name,
            session_id=state.get("session_id", ""),
            experiment_output=experiment_output,  # 실험용 구조화된 출력
        )

    return node_function
