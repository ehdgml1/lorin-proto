"""
Planner Node Module - Planning (Question Decomposition)

- planning_node: Decomposes user questions into executable subquery sets
  - Output: 'Planning Results (PLANNING PREPROCESS)' block + JSON structure (subqueries, notes, assumptions)
  - Policy: Anchor-first (Q1), chained depends_on, flexible count (default 2-6, allows 1 for simple cases)
  - NOTE: This module only generates plans and stores them in metadata.planner.
    Answer generation for subqueries is handled by llm_node.
"""

from __future__ import annotations
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from ...llm import Chatbot
from ...logger.logger import get_logger
from ..state import AgentState
from ..schema import MetadataManager
from ...prompt.template_loader import PromptTemplateLoader

logger = get_logger(__name__)

def _keep_from_planning_header(text: str) -> str:
    """Preserve text from 'Planning Results (PLANNING PREPROCESS)' header onwards"""
    if not text:
        return text

    # Support both Korean and English headers
    headers = [
        "분해 결과(PLANNING PREPROCESS)",
        "Planning Results (PLANNING PREPROCESS)",
        "Planning Results (ACTIVITY DETECTION)"
    ]

    for hdr in headers:
        i = text.find(hdr)
        if i >= 0:
            return text[i:].strip()

    return text.strip()
# --- Small Utilities ---
def _clip(s: Any, n: int = 160) -> str:
    try:
        s = str(s)
    except Exception:
        s = repr(s)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"

# --- Environment Configuration (with sensible defaults) ---
def _get_env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        if v is None:
            return default
        return max(0, int(v))
    except Exception:
        return default

def _get_env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")

def _assumptions_line() -> str:
    # Project fixed assumptions (can be overridden via .env)
    v = os.getenv(
        "ASSUMPTIONS_LINE",
        "Environment: Android OS *logcat logs* (only component + message exist)"
        "Purpose: Query decomposition for finding relevant logs (debugging) to resolve user question issues"
    )
    return v.strip()

def _subq_bounds() -> tuple[int, int]:
    lo = _get_env_int("PLANNER_SUBQ_MIN", 1)
    hi = _get_env_int("PLANNER_SUBQ_MAX", 5)
    if lo < 1:
        lo = 1
    if hi < lo:
        hi = lo
    return lo, hi

def _force_anchor() -> bool:
    return _get_env_bool("PLANNER_FORCE_ANCHOR", True)

def _language() -> str:
    return (os.getenv("PLANNER_LANGUAGE") or "ko").lower()

# --- Message/Text Utilities ---
def _ak(m: BaseMessage) -> dict:
    return (getattr(m, "additional_kwargs", {}) or getattr(m, "kwargs", {}) or {})

def _get_agent_name(m: BaseMessage) -> str:
    ak = _ak(m)
    if "agent_name" in ak:
        return ak["agent_name"]
    from langchain_core.messages import HumanMessage, AIMessage
    if isinstance(m, HumanMessage):
        return "human"
    if isinstance(m, AIMessage):
        return "ai"
    return m.__class__.__name__

def _last_human_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = getattr(m, "content", "")
            if c:
                return c if isinstance(c, str) else str(c)
    return ""

def _normalize_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        parts = []
        for p in obj:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(obj)

# --- Prompt Builder ---
def _build_planner_prompt(user_q: str, lo: int, hi: int, force_anchor: bool, lang: str, intent: str = "debug", mode: str = "parallel", decomposition: dict = None) -> str:
    """Build planner prompt using Jinja2 template with intent-based template selection

    Args:
        user_q: User question
        lo: Minimum number of subqueries
        hi: Maximum number of subqueries
        force_anchor: Force anchor type for first subquery
        lang: Language (currently unused but kept for compatibility)
        intent: Query intent - "debug" or "analysis" (default: "debug")
        mode: Planning mode - "parallel" (default), "initial", or "expansion"
        decomposition: Optional question decomposition with primary_issue, primary_trigger, context_scope

    Returns:
        str: Rendered prompt from appropriate template
    """
    template_loader = PromptTemplateLoader()

    # Select template based on mode first, then intent
    if mode == "initial":
        template_name = "planner_initial.j2"
        prompt = template_loader.render_template(
            template_name,
            user_question=user_q,
            decomposition=decomposition
        )
        logger.info(f"[_build_prompt] Generated prompt: {len(prompt)} chars, {len(prompt.split())} words")
        logger.debug(f"[_build_prompt] Prompt preview (first 500 chars):\n{prompt[:500]}")
        return prompt
    elif mode == "expansion":
        # This should not be called directly - use _build_expansion_prompt instead
        raise ValueError("Use _build_expansion_prompt for expansion mode")
    else:  # parallel mode (default)
        # Select template based on intent
        if intent == "analysis":
            template_name = "planner_activity.j2"
        else:  # debug (default)
            template_name = "planner_tot.j2"

        return template_loader.render_template(
            template_name,
            user_question=user_q,
            min_queries=lo,
            max_queries=hi,
            assumptions_line=_assumptions_line(),
            force_anchor=force_anchor,
            decomposition=decomposition
        )


def _build_expansion_prompt(user_q: str, pivot_doc: dict, decomposition: dict = None) -> str:
    """Build expansion prompt for Round 2 based on pivot document

    LLM determines all temporal ranges dynamically based on pivot analysis.
    No hardcoded ranges - the model analyzes pivot content and decides:
    - How many queries needed (2-6)
    - What temporal range each query should cover
    - What aspect each query investigates

    Args:
        user_q: Original user question
        pivot_doc: Pivot document with position and content
        decomposition: Optional question decomposition with primary_issue, primary_trigger, context_scope

    Returns:
        str: Rendered expansion prompt
    """
    template_loader = PromptTemplateLoader()

    pivot_position = pivot_doc.get("position", 0.5)
    pivot_content = pivot_doc.get("content", "")

    # Extract total_lines from pivot metadata for adaptive range sizing
    pivot_metadata = pivot_doc.get("metadata", {})
    total_lines = pivot_metadata.get("total_lines", 5000)  # Default 5K if not available

    logger.info(f"[expansion_prompt] Log size: {total_lines} lines | pivot position: {pivot_position:.4f}")

    # LLM determines ranges dynamically based on log size
    # Clear template cache to ensure latest template is used
    template_loader.clear_cache()

    return template_loader.render_template(
        "planner_expansion.j2",
        user_question=user_q,
        pivot_position=pivot_position,
        pivot_content=pivot_content,
        total_lines=total_lines,
        decomposition=decomposition
    )


def _validate_expansion_ranges(expansion_queries: list, pivot_position: float) -> bool:
    """Validate dynamic expansion queries (2-6 queries, overlaps allowed)

    Generic validation with no case-specific logic:
    - Check query count (2-6)
    - Verify position_filter format and bounds
    - Warn on very narrow ranges

    Args:
        expansion_queries: List of expansion query dicts
        pivot_position: Pivot position for reference logging

    Returns:
        bool: True if all queries valid
    """
    n_queries = len(expansion_queries)

    # Check query count (minimum 3 queries required)
    if not (3 <= n_queries <= 6):
        logger.error(f"[validation] Invalid query count: {n_queries} (expected 3-6)")
        return False

    logger.info(f"[validation] Validating {n_queries} dynamic expansion queries")

    # Collect and validate ranges
    ranges = []
    for i, query in enumerate(expansion_queries):
        qid = query.get("id", f"Q{i+1}")
        position_filter = query.get("position_filter", [])

        if not isinstance(position_filter, list) or len(position_filter) != 2:
            logger.error(f"[validation] {qid}: Invalid position_filter format (expected [start, end])")
            return False

        start, end = position_filter

        # Basic validation: bounds and ordering
        if not (0.0 <= start < end <= 1.0):
            logger.error(f"[validation] {qid}: Invalid range [{start:.4f}, {end:.4f}] (must be 0.0 ≤ start < end ≤ 1.0)")
            return False

        # Warn on very narrow ranges (< 1%) or very broad ranges (> 30%)
        range_width = end - start
        if range_width < 0.01:
            logger.warning(f"[validation] {qid}: Very narrow range [{start:.4f}, {end:.4f}] ({range_width*100:.1f}%)")
        elif range_width > 0.30:
            logger.warning(f"[validation] {qid}: Very broad range [{start:.4f}, {end:.4f}] ({range_width*100:.1f}%) - may be inefficient")

        ranges.append((start, end, qid))

    # Calculate total coverage (may exceed 100% with overlaps)
    total_coverage = sum(end - start for start, end, _ in ranges)
    logger.info(f"[validation] Total temporal coverage: {total_coverage*100:.1f}% (overlaps allowed)")

    if total_coverage < 0.10:
        logger.warning("[validation] Low coverage (<10%), might miss important events")
    elif total_coverage > 1.5:
        logger.info(f"[validation] High overlap detected (coverage: {total_coverage*100:.1f}%)")

    logger.info("All ranges valid")
    return True


# --- JSON Extraction/Validation (Light Guard) ---

def _extract_first_json_block(text: str) -> Optional[dict]:
    """Extract JSON - uses unified utility"""
    from ...utils.json_parser import extract_json_from_response
    return extract_json_from_response(text)

def _guard_fix_plan(plan: dict) -> dict:
    """Simple chain/ID validation fix: Q1..Qk ordering and depends_on chaining"""
    if not isinstance(plan, dict):
        return plan
    subqs = plan.get("subqueries")
    if not isinstance(subqs, list) or not subqs:
        return plan

    fixed: List[dict] = []
    # Rearrange IDs: Q1..Qk
    for i, sq in enumerate(subqs, 1):
        sq = dict(sq or {})
        sq["id"] = f"Q{i}"
        # Force depends_on chain
        if i == 1:
            sq["depends_on"] = []
            sq.setdefault("type", "anchor")  # Set Q1 as anchor if possible
        else:
            sq["depends_on"] = [f"Q{i-1}"]
        # Default intent protection
        if sq.get("intent") not in {"retrieve", "filter", "correlate", "compute", "summarize"}:
            sq["intent"] = "retrieve" if i == 1 else "filter"
        # Field augmentation
        sq.setdefault("input_querry", "")
        sq.setdefault("output", "")
        fixed.append(sq)

    plan["subqueries"] = fixed
    plan["depends_mode"] = "chain"
    plan.setdefault("notes", [])
    plan.setdefault("assumptions", [])
    return plan

# --- Pivot Extraction Helper ---
async def select_best_pivot_with_llm(
    candidates: List[dict],
    initial_query: dict,
    chatbot: Chatbot,
    primary_issue: str = None
) -> Optional[dict]:
    """Use LLM to select the most relevant pivot from multiple candidates

    Args:
        candidates: List of pivot candidates with score, position, content
        initial_query: Initial query with text, expected_region, reasoning
        chatbot: Chatbot instance for LLM calls
        primary_issue: Optional primary issue from question decomposition

    Returns:
        Selected pivot dict or None if selection fails
    """
    if not candidates:
        logger.error("[pivot_selection] No candidates provided")
        return None

    if len(candidates) == 1:
        logger.info("[pivot_selection] Only 1 candidate - auto-selecting")
        return candidates[0]

    logger.info(f"[pivot_selection] Selecting from {len(candidates)} candidates using LLM")

    # Prepare template data
    template_loader = PromptTemplateLoader()

    # Take top 3 candidates
    top_candidates = candidates[:3]

    initial_query_text = initial_query.get("text", "")
    expected_region = initial_query.get("expected_region", "unknown")
    initial_reasoning = initial_query.get("reasoning", "")

    # Extract total_lines from first candidate's metadata for adaptive temporal zones
    total_lines = 5000  # Default fallback
    if top_candidates and top_candidates[0].get("metadata"):
        total_lines = top_candidates[0]["metadata"].get("total_lines", 5000)

    logger.info(f"[pivot_selection] Log size: {total_lines} lines for adaptive temporal zones")

    try:
        # Render validation prompt (initial_query-centric, with adaptive temporal zones)
        prompt = template_loader.render_template(
            "pivot_validation.j2",
            initial_query_text=initial_query_text,
            expected_region=expected_region,
            initial_reasoning=initial_reasoning,
            candidates=top_candidates,
            total_lines=total_lines,
            primary_issue=primary_issue
        )

        logger.info(f"[pivot_selection] Rendered prompt length: {len(prompt)} chars")
        logger.info(f"[pivot_selection] Prompt preview (first 500 chars):\n{prompt[:500]}")
        logger.info(f"[pivot_selection] Chatbot type: {type(chatbot)}")
        logger.info("[pivot_selection] Calling LLM for pivot validation")

        res = await chatbot.ask(prompt)
        logger.info(f"[pivot_selection] Response type: {type(res)}")
        logger.info(f"[pivot_selection] Response object: {res}")

        raw = _normalize_text(getattr(res, "content", res)).strip()

        logger.info(f"[pivot_selection] LLM response length: {len(raw)} chars")
        if raw:
            logger.info(f"[pivot_selection] LLM response preview (first 500 chars):\n{raw[:500]}")
        else:
            logger.error("[pivot_selection] LLM returned empty response!")
            logger.error(f"[pivot_selection] res.content = {getattr(res, 'content', 'NO CONTENT ATTR')}")

        # Extract JSON from response
        selection_json = _extract_first_json_block(raw)

        if not selection_json:
            logger.error("[pivot_selection] Failed to extract JSON from LLM response")
            logger.warning("[pivot_selection] Falling back to score-based selection (Top-1)")
            return candidates[0]

        # Get selected candidate index
        selected_idx = selection_json.get("selected_candidate")
        reasoning = selection_json.get("reasoning", "N/A")
        temporal_analysis = selection_json.get("temporal_analysis", "N/A")

        if not isinstance(selected_idx, int) or not (1 <= selected_idx <= len(top_candidates)):
            logger.error(f"[pivot_selection] Invalid candidate index: {selected_idx}")
            logger.warning("[pivot_selection] Falling back to score-based selection (Top-1)")
            return candidates[0]

        # Convert 1-indexed to 0-indexed
        selected_pivot = top_candidates[selected_idx - 1]

        logger.info(
            "[pivot_selection] LLM selected Candidate %d | position=%.4f, score=%.4f",
            selected_idx, selected_pivot["position"], selected_pivot["score"]
        )
        logger.info(f"[pivot_selection] Reasoning: {_clip(reasoning, 200)}")
        logger.info(f"[pivot_selection] Temporal analysis: {_clip(temporal_analysis, 200)}")

        # Log rejected candidates
        rejected_reasons = selection_json.get("rejected_reasons", {})
        for idx_str, reason in rejected_reasons.items():
            logger.info(f"[pivot_selection] Rejected Candidate {idx_str}: {_clip(reason, 100)}")

        return selected_pivot

    except Exception as e:
        logger.error(f"[pivot_selection] Exception during LLM selection: {str(e)}")
        logger.warning("[pivot_selection] Falling back to score-based selection (Top-1)")
        return candidates[0]


def extract_pivot_from_retrieval(state: AgentState, return_top_n: int = 1) -> Optional[dict]:
    """Extract top N pivot candidates from initial retrieval results

    Uses MetadataManager for safe extraction of pivot from search results.

    Args:
        state: Current agent state with retrieval results
        return_top_n: Number of top candidates to return (default 1 for backward compat)

    Returns:
        dict with pivot info (single) or list of dicts (multiple) or None if no results found
    """
    # Safe metadata lookup using MetadataManager
    faiss_retriever = MetadataManager.safe_get_metadata(state, "faiss_retriever")

    if not faiss_retriever:
        logger.warning("[pivot_extraction] No faiss_retriever metadata found")
        return None

    # Get results_by_qid (dict structure from memory_optimized_retrieve_node)
    results_by_qid = faiss_retriever.get("results_by_qid", {})

    if not results_by_qid:
        logger.warning("[pivot_extraction] No retrieval results found in metadata['faiss_retriever']['results_by_qid']")
        logger.warning(f"[pivot_extraction] Available keys in faiss_retriever: {list(faiss_retriever.keys())}")
        return None

    logger.info(f"[pivot_extraction] Found {len(results_by_qid)} query results")

    # Extract all documents from all queries
    all_docs = []
    for qid, result_data in results_by_qid.items():
        if not isinstance(result_data, dict):
            logger.warning(f"[pivot_extraction] QID {qid}: unexpected result type {type(result_data)}")
            continue

        # Get results list from this query
        results = result_data.get("results", [])
        logger.info(f"[pivot_extraction] QID {qid}: {len(results)} results")

        for doc in results:
            # Extract score and metadata (doc is already a dict from asdict conversion)
            if isinstance(doc, dict):
                score = doc.get("score", 0.0)
                content = doc.get("content", "")
                doc_metadata = doc.get("metadata", {})
            else:
                logger.warning(f"[pivot_extraction] QID {qid}: unexpected doc type {type(doc)}")
                score = getattr(doc, "score", 0.0)
                content = getattr(doc, "page_content", "") or getattr(doc, "content", "")
                doc_metadata = getattr(doc, "metadata", {})

            all_docs.append({
                "score": score,
                "content": content,
                "position": doc_metadata.get("relative_position", 0.5),
                "line_number": doc_metadata.get("line_number", 0),
                "metadata": doc_metadata
            })

    if not all_docs:
        logger.warning("[pivot_extraction] No documents found in retrieval results")
        return None

    # Sort by score descending
    all_docs.sort(key=lambda x: x["score"], reverse=True)
    
    if return_top_n == 1:
        # Backward compatible: return single pivot
        pivot = all_docs[0]
        logger.info(
            "[pivot_extraction] Pivot selected | position=%.4f, score=%.4f, line=%d",
            pivot["position"], pivot["score"], pivot["line_number"]
        )
        logger.info("[pivot_extraction] Pivot content preview: %s", _clip(pivot["content"], 200))

        return {
            "position": pivot["position"],
            "content": pivot["content"],
            "line_number": pivot["line_number"],
            "score": pivot["score"],
            "metadata": pivot["metadata"]
        }
    else:
        # Return top N candidates for LLM selection
        candidates = all_docs[:return_top_n]
        logger.info(f"[pivot_extraction] Extracted {len(candidates)} pivot candidates for validation")
        
        for i, cand in enumerate(candidates, 1):
            logger.info(
                f"[pivot_extraction] Candidate {i} | position=%.4f, score=%.4f, line=%d",
                cand["position"], cand["score"], cand["line_number"]
            )
        
        return candidates


# --- Planner Node ---

def planning_node(
    chatbot: Optional[Chatbot] = None,
    agent_name: str = "planner",
    *,
    retries: int = 3,
    mode: str = "parallel",  # "parallel", "initial", or "expansion"
) -> Callable[[AgentState], AgentState]:

    if chatbot is None:
        from ...utils import create_chatbot_from_env
        chatbot = create_chatbot_from_env(
            temperature=0.4
            # max_tokens uses default (4096)
        )

    async def node_function(state: AgentState) -> AgentState:
        messages: List[BaseMessage] = state.get("messages", []) or []
        metadata: Dict[str, Any] = state.get("metadata", {}) or {}

        # Get latest user utterance
        user_q = _last_human_text(messages)
        if not user_q:
            md = metadata.copy()
            md.setdefault("planner", {})["status"] = "no_user_input"
            logger.info("[planner] no_user_input -> skip")
            return type(state)(**{**state, "metadata": md})

        logger.info("[planner] start | last_human='%s'", _clip(user_q, 180))

        # Question Decomposition (parse structure before intent classification)
        from .question_decomposer import decompose_question

        try:
            decomposition = await decompose_question(
                user_question=user_q,
                chatbot=chatbot
            )

            # Validate decomposition is a dict
            if not isinstance(decomposition, dict):
                logger.error(f"[planner] decompose_question returned non-dict: {type(decomposition)}")
                logger.error(f"[planner] decomposition value: {decomposition}")
                # Create safe default
                decomposition = {
                    "primary_issue": user_q,
                    "primary_trigger": "unknown",
                    "secondary_triggers": [],
                    "context_scope": "unknown",
                    "temporal_hint": "unknown",
                    "reasoning": "Decomposition returned invalid type"
                }

            logger.info(f"[planner] Question decomposed:")
            logger.info(f"  Primary Issue: {decomposition.get('primary_issue', 'N/A')}")
            logger.info(f"  Primary Trigger: {decomposition.get('primary_trigger', 'N/A')}")
            logger.info(f"  Context Scope: {decomposition.get('context_scope', 'N/A')}")
            logger.info(f"  Temporal Hint: {decomposition.get('temporal_hint', 'N/A')}")

        except Exception as e:
            logger.error(f"[planner] Question decomposition failed: {e}")
            logger.error(f"[planner] Exception type: {type(e).__name__}")
            # Use safe defaults
            decomposition = {
                "primary_issue": user_q,
                "primary_trigger": "unknown",
                "secondary_triggers": [],
                "context_scope": "unknown",
                "temporal_hint": "unknown",
                "reasoning": f"Decomposition failed: {e}"
            }
            logger.info("[planner] Using safe default decomposition")

        # Store decomposition in metadata for downstream use
        metadata = metadata.copy()
        metadata["question_decomposition"] = decomposition

        # LLM-based intent classification (non-heuristic, with decomposition context)
        from .intent_classifier import determine_intent

        intent = await determine_intent(
            user_question=user_q,
            metadata=metadata,  # Now includes decomposition
            chatbot=chatbot,
            force_llm=False  # Allow metadata override if present
        )

        logger.info(f"[planner] Detected intent: '{intent}' for question: '{_clip(user_q, 100)}'")

        # Check if we're in expansion mode (Round 2)
        # Safe planner metadata lookup using MetadataManager
        existing_planner_data = MetadataManager.safe_get_metadata(state, "planner") or {}
        current_mode = existing_planner_data.get("phase", mode)

        # Plan generation (passes intent + mode) with JSON validation retry
        lo, hi = _subq_bounds()
        force_anchor = _force_anchor()

        # Build prompt based on current mode
        if current_mode == "expansion":
            # Round 2: Expansion mode - need pivot from metadata
            pivot_doc = existing_planner_data.get("pivot")
            if not pivot_doc:
                logger.error("[planner] expansion mode but no pivot found in metadata")
                # Fallback to initial mode
                current_mode = "initial"
                prompt = _build_planner_prompt(user_q, lo, hi, force_anchor, _language(), intent=intent, mode="initial", decomposition=decomposition)
            else:
                logger.info(f"[planner] Building expansion prompt with pivot at position {pivot_doc.get('position', 'N/A')}")
                prompt = _build_expansion_prompt(user_q, pivot_doc, decomposition=decomposition)

                # Log decomposition context in expansion prompt
                logger.info(f"[planner] Decomposition passed to expansion prompt:")
                logger.info(f"  - primary_trigger: {decomposition.get('primary_trigger', 'N/A')}")
                logger.info(f"  - context_scope: {decomposition.get('context_scope', 'N/A')}")

                # Log full prompt length and check for Context Reminder
                logger.info(f"[planner] Full expansion prompt length: {len(prompt)} chars")
                context_reminder_present = "# Context Reminder" in prompt
                logger.info(f"[planner] Context Reminder present in prompt: {context_reminder_present}")

                # Log actual rendered prompt snippet (first 3500 chars to see if decomposition appears after pivot content)
                logger.info(f"[planner] Expansion prompt preview (first 3500 chars):\n{prompt[:3500]}")
        elif current_mode == "initial":
            # Round 1: Initial mode - generate single focused query
            logger.info("[planner] Building initial query prompt (Round 1)")
            prompt = _build_planner_prompt(user_q, lo, hi, force_anchor, _language(), intent=intent, mode="initial", decomposition=decomposition)
        else:
            # Default: Parallel mode (backward compatible)
            prompt = _build_planner_prompt(user_q, lo, hi, force_anchor, _language(), intent=intent, mode="parallel", decomposition=decomposition)

        content = ""
        plan_json = None
        err_msg = None

        # Retry up to 3 times for both LLM call AND JSON parsing
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[planner] Attempt {attempt}/{max_attempts}: Calling LLM for planning")
                res = await chatbot.ask(prompt)
                raw = _normalize_text(getattr(res, "content", res)).strip()

                # DEBUG: Log actual response from Gemini
                logger.info(f"[planner] Attempt {attempt}/{max_attempts}: Raw response length: {len(raw)} chars")
                logger.info(f"[planner] Attempt {attempt}/{max_attempts}: Raw response preview (first 500 chars):\n{raw[:500]}")
                if len(raw) > 500:
                    logger.info(f"[planner] Attempt {attempt}/{max_attempts}: Raw response end (last 300 chars):\n{raw[-300:]}")

                # Sequential mode: Skip header check, use raw JSON response
                if current_mode in ("initial", "expansion"):
                    # Sequential mode templates return pure JSON without headers
                    content = raw.strip()
                    logger.info(f"[planner] Sequential mode ({current_mode}): Using raw response (no header check)")
                else:
                    # Parallel mode: Extract from planning header
                    content = _keep_from_planning_header(raw)

                    # Check for planning headers (parallel mode only)
                    if not content or not any(h in content for h in ["분해 결과(PLANNING PREPROCESS)", "Planning Results"]):
                        logger.warning(f"[planner] Attempt {attempt}/{max_attempts}: No planning header found in response")
                        logger.warning(f"[planner] Attempt {attempt}/{max_attempts}: Content after header extraction: '{content[:200] if content else '(empty)'}'")
                        if attempt < max_attempts:
                            continue
                        else:
                            # Even without header, try to proceed with raw content
                            content = raw
                            logger.info(f"[planner] Attempt {attempt}/{max_attempts}: Using raw response as fallback")
                            break

                # Try to extract JSON
                logger.info(f"[planner] Attempt {attempt}/{max_attempts}: Attempting JSON extraction from content ({len(content)} chars)")
                plan_json = _extract_first_json_block(content)

                if plan_json:
                    logger.info(f"[planner] Attempt {attempt}/{max_attempts}: JSON extraction successful")
                    logger.info(f"[planner] Attempt {attempt}/{max_attempts}: Extracted JSON keys: {list(plan_json.keys())}")
                    break  # Success - exit retry loop
                else:
                    logger.warning(f"[planner] Attempt {attempt}/{max_attempts}: JSON extraction failed")
                    # DEBUG: Show what we tried to parse
                    logger.warning(f"[planner] Attempt {attempt}/{max_attempts}: Failed to extract JSON from:\n{content[:1000]}")
                    if attempt < max_attempts:
                        logger.info(f"[planner] Retrying with fresh LLM call...")
                        continue

            except Exception as e:
                err_msg = str(e)
                logger.error(f"[planner] Attempt {attempt}/{max_attempts}: Exception - {err_msg}")
                if attempt < max_attempts:
                    logger.info(f"[planner] Retrying after exception...")
                    continue

        # Log final result
        if plan_json:
            logger.info(f"[planner] Successfully extracted JSON after {attempt} attempt(s)")
        else:
            logger.error(f"[planner] Failed to extract JSON after {max_attempts} attempts")

        ts = datetime.now(timezone.utc).isoformat()
        plan_msg = AIMessage(
            content=content or "(플래너 응답 없음)",
            additional_kwargs={"agent_name": agent_name, "timestamp": ts},
        )
        new_messages = messages + [plan_msg]
        logger.info("[planner] raw_plan_len=%d | preview=%s", len(plan_msg.content or ""), _clip(plan_msg.content, 140))

        # JSON 보정
        logger.info("[planner] plan_json_extracted=%s", bool(plan_json))
        fixed_json = _guard_fix_plan(plan_json) if plan_json else None
        logger.info("[planner] plan_json_fixed=%s", bool(fixed_json))

        guard_msg = None
        if fixed_json and fixed_json != plan_json:
            guard_text = (
                "분해 결과(PLANNING PREPROCESS)\n"
                "[교정] planner 출력의 JSON에서 ID/depends_on 체인을 보정했습니다.\n"
                "[하위질의(JSON)]\n" + json.dumps(fixed_json, ensure_ascii=False, indent=2)
            )
            guard_msg = AIMessage(
                content=guard_text,
                additional_kwargs={"agent_name": "planner-guard", "timestamp": ts},
            )
            new_messages.append(guard_msg)
            logger.info("[planner] guard_msg appended")

        # Metadata update (NOTE: not using plan_executor)
        md = metadata.copy()

        # Increment planner iterations (experimental metric)
        planner_iterations = md.get("planner_iterations", 0) + 1
        md["planner_iterations"] = planner_iterations
        logger.info(f"[planner] Planner iteration: {planner_iterations}")

        pns = md.get("planner", {}) or {}
        plan_core = fixed_json or plan_json or {}
        subqs = (plan_core.get("subqueries") or [])

        # Extract temporal_context from the plan (new field from improved prompt)
        temporal_ctx = plan_core.get("temporal_context", {})

        pns["temporal_context"] = temporal_ctx  # Store temporal context for retriever
        pns["intent"] = intent  # Store detected intent
        pns["phase"] = current_mode  # Store current phase for workflow control

        # Handle phase-specific logic
        if current_mode == "initial":
            # Round 1: Initial query generated
            # Extract single query from initial_query field
            initial_q = plan_core.get("initial_query", {})

            # Fallback: If LLM omitted "initial_query" wrapper, use plan_core directly
            if not initial_q and "text" in plan_core:
                logger.warning("[planner] LLM omitted 'initial_query' wrapper, using root fields as fallback")
                initial_q = {
                    "text": plan_core.get("text", ""),
                    "expected_region": plan_core.get("expected_region", "unknown"),
                    "reasoning": plan_core.get("reasoning", "")
                }

            if initial_q and initial_q.get("text"):
                # Convert to subquery format for retrieval
                initial_subq = {
                    "id": "Q0",
                    "text": initial_q.get("text", ""),
                    "type": "initial",
                    "expected_region": initial_q.get("expected_region", "unknown"),
                    "reasoning": initial_q.get("reasoning", "")
                }
                pns["initial_query"] = initial_subq
                # Update subqueries to contain just this one query
                plan_core["subqueries"] = [initial_subq]
                subqs = [initial_subq]
                logger.info("[planner] Round 1 complete: Initial query generated - '%s'", _clip(initial_subq["text"], 100))
                # Mark that we need to move to expansion phase after retrieval
                pns["needs_expansion"] = True
            else:
                logger.error("[planner] initial mode but no initial_query found in plan_core")
                logger.error(f"[planner] plan_core keys: {list(plan_core.keys())}")
                logger.error(f"[planner] plan_core content: {json.dumps(plan_core, ensure_ascii=False, indent=2)[:1000]}")
        elif current_mode == "expansion":
            # Round 2: Expansion queries generated
            # These are now the "subqueries" for downstream processing
            expansion_qs = plan_core.get("expansion_queries", [])

            # Fallback: If LLM put queries in "queries" or "subqueries" instead
            if not expansion_qs:
                if "queries" in plan_core:
                    logger.warning("[planner] LLM used 'queries' instead of 'expansion_queries', using as fallback")
                    expansion_qs = plan_core.get("queries", [])
                elif "subqueries" in plan_core and plan_core["subqueries"]:
                    # Check if subqueries are already expansion queries (have position_filter)
                    first_sq = plan_core["subqueries"][0] if plan_core["subqueries"] else {}
                    if "position_filter" in first_sq:
                        logger.warning("[planner] LLM put expansion queries in 'subqueries' directly, using as fallback")
                        expansion_qs = plan_core.get("subqueries", [])

            if expansion_qs:
                # Validate LLM-generated dynamic ranges
                # existing_planner_data already queried via MetadataManager (line 650)
                pivot_position = existing_planner_data.get("pivot", {}).get("position", 0.5)

                if not _validate_expansion_ranges(expansion_qs, pivot_position):
                    logger.error("[planner] Invalid expansion ranges generated by LLM")
                    logger.error("[planner] Rejecting plan - LLM will retry on next attempt")
                    # Set empty queries to trigger retry
                    expansion_qs = []
                else:
                    # Log query strategy
                    query_strategy = plan_core.get("query_strategy", "N/A")
                    logger.info(f"[planner] Query strategy: {query_strategy}")

                    # Log each query with details
                    for q in expansion_qs:
                        qid = q.get("id")
                        qtype = q.get("type", "unknown")
                        pos_filter = q.get("position_filter", [])
                        reasoning = q.get("reasoning", "")
                        text = q.get("text", "")
                        logger.info(
                            f"[planner] {qid} ({qtype}): [{pos_filter[0]:.4f}, {pos_filter[1]:.4f}]"
                        )
                        logger.info(f"[planner]   Text: {_clip(text, 100)}")
                        logger.info(f"[planner]   Reasoning: {_clip(reasoning, 150)}")

            if expansion_qs:
                # Update subqueries to be the expansion queries
                plan_core["subqueries"] = expansion_qs
                subqs = expansion_qs
                logger.info("[planner] Round 2 complete: %d dynamic expansion queries validated", len(expansion_qs))
                pns["needs_expansion"] = False
            else:
                logger.error("[planner] expansion mode but no valid expansion_queries")
                logger.error(f"[planner] plan_core keys: {list(plan_core.keys())}")
                if plan_core:
                    logger.error(f"[planner] plan_core content: {json.dumps(plan_core, ensure_ascii=False, indent=2)[:1000]}")

        # Always set n_subqueries after phase-specific logic (ensures no KeyError)
        pns["n_subqueries"] = len(subqs)

        # Save plan_core AFTER phase-specific logic has updated subqueries
        pns["last_plan_json"] = plan_core

        # Log temporal context for debugging
        if temporal_ctx:
            logger.info(
                "[planner] temporal_context detected | region=%s, reasoning=%s",
                temporal_ctx.get("region", "N/A"),
                _clip(temporal_ctx.get("reasoning", ""), 100)
            )
        plan_sig = None
        try:
            plan_sig = hashlib.sha256(
                json.dumps(plan_core, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
        except Exception:
            pass

        # Reset if plan changed (different signature) or classifier progress doesn't match plan
        prev_cd = (md.get("classifier") or {})
        prev_sig = prev_cd.get("plan_sig")
        prev_progress = prev_cd.get("progress") or {}
        need_reset = False

        if plan_sig and plan_sig != prev_sig:
            need_reset = True
        elif (prev_progress.get("total") != len(subqs)) or (prev_progress.get("idx", 0) > len(subqs)):
            need_reset = True

        if need_reset:
            logger.info(
                "[planner] detected new/changed plan -> reset classifier domain | old_sig=%s new_sig=%s",
                prev_sig, plan_sig
            )
        else:
            # Only update total count to latest (safe)
            if "progress" in prev_cd:
                prev_cd["progress"]["total"] = len(subqs)
                md["classifier"] = prev_cd
        if subqs:
            # Log subqueries with type and temporal filter
            lines = []
            for sq in subqs:
                sq_id = sq.get('id', '?')
                sq_text = _clip(sq.get('text', ''), 120)
                sq_type = sq.get('type', 'N/A')

                # Check for position_filter (expansion queries) or show N/A
                position_filter = sq.get('position_filter')
                if position_filter and isinstance(position_filter, list) and len(position_filter) == 2:
                    sq_temporal = f"[{position_filter[0]:.4f}, {position_filter[1]:.4f}]"
                else:
                    sq_temporal = 'N/A'

                lines.append(f"- {sq_id} [{sq_type}] {sq_text}")
                lines.append(f"  Temporal: {sq_temporal}")
            logger.info("[planner] questions:\n%s", "\n".join(lines))

        pns["depends_mode"] = plan_core.get("depends_mode", "chain")
        pns["has_anchor"] = any((sq.get("type") == "anchor") for sq in subqs)
        if err_msg:
            pns["error"] = err_msg
        md["planner"] = pns
        md["last_agent"] = "planner-guard" if guard_msg else agent_name

        logger.info(
            "[planner] plan saved | n_subqueries=%d, has_anchor=%s, keys=%s",
            pns["n_subqueries"], pns["has_anchor"], list(pns.keys())
        )
        if subqs:
            logger.debug("[planner] first_subq=%s", _clip(subqs[0].get("text", "")))

        # Don't touch context (summarizer/llm_node handles context management)
        return type(state)(
            messages=new_messages,
            context_messages=state.get("context_messages", []),
            metadata=md,
            current_agent=md["last_agent"],
            session_id=state.get("session_id", ""),
        )

    return node_function


# --- Pivot Transition Node (for sequential workflow) ---

def pivot_transition_node(
    chatbot: Optional[Chatbot] = None
) -> Callable[[AgentState], AgentState]:
    """Node function to extract pivot after initial retrieval and prepare for expansion

    This node:
    1. Extracts Top-3 pivot candidates from initial retrieval results
    2. Uses LLM to select the most relevant pivot based on initial query intent
    3. Stores selected pivot in metadata
    4. Sets phase to "expansion" for next planner call

    Args:
        chatbot: Optional chatbot instance for LLM-based pivot validation

    Returns:
        Callable: Async node function
    """
    if chatbot is None:
        from ...utils import create_chatbot_from_env
        # Use sufficient max_tokens for JSON response with reasoning
        chatbot = create_chatbot_from_env(temperature=0.3, max_tokens=4096)

    async def node_function(state: AgentState) -> AgentState:
        logger.info("[pivot_transition] Starting pivot extraction and validation")

        metadata = state.get("metadata", {}) or {}
        # Safe planner metadata lookup using MetadataManager
        planner_data = MetadataManager.safe_get_metadata(state, "planner") or {}
        messages = state.get("messages", [])

        # Check if we need pivot extraction (only if in initial phase)
        phase = planner_data.get("phase", "parallel")
        if phase != "initial":
            logger.info(f"[pivot_transition] Not in initial phase (current: {phase}), skipping")
            return state

        # Get initial query from planner metadata (primary selection criteria)
        initial_query = planner_data.get("initial_query", {})
        if not initial_query:
            logger.warning("[pivot_transition] No initial_query found in planner metadata")
            logger.warning("[pivot_transition] This may indicate a workflow issue")

        # Extract primary_issue from decomposition for pivot selection context
        # Safe question_decomposition metadata lookup using MetadataManager
        decomposition = MetadataManager.safe_get_metadata(state, "question_decomposition") or {}
        primary_issue = decomposition.get("primary_issue")
        logger.info(f"[pivot_transition] Primary issue for pivot selection: {primary_issue or 'N/A'}")

        # Extract Top-3 pivot candidates
        candidates = extract_pivot_from_retrieval(state, return_top_n=3)

        if not candidates:
            logger.error("[pivot_transition] Failed to extract pivot candidates - cannot proceed to expansion")
            # Mark as error
            md = metadata.copy()
            planner_ns = md.get("planner", {}).copy()
            planner_ns["pivot_extraction_failed"] = True
            planner_ns["needs_expansion"] = False
            md["planner"] = planner_ns
            return type(state)(**{**state, "metadata": md})

        # Ensure candidates is a list (handle backward compat case)
        if isinstance(candidates, dict):
            # Single candidate returned - use directly
            pivot = candidates
            logger.info("[pivot_transition] Single candidate mode - using directly")
        else:
            # Multiple candidates - use LLM to select best one (initial_query-centric)
            logger.info(f"[pivot_transition] Using LLM to select from {len(candidates)} candidates")
            logger.info(f"[pivot_transition] Selection criteria: Initial query intent alignment")

            try:
                pivot = await select_best_pivot_with_llm(
                    candidates=candidates,
                    initial_query=initial_query,
                    chatbot=chatbot,
                    primary_issue=primary_issue
                )
            except Exception as e:
                logger.error(f"[pivot_transition] LLM selection failed: {str(e)}")
                logger.warning("[pivot_transition] Falling back to score-based Top-1")
                pivot = candidates[0] if isinstance(candidates, list) else candidates

        if not pivot:
            logger.error("[pivot_transition] No pivot selected - cannot proceed to expansion")
            md = metadata.copy()
            planner_ns = md.get("planner", {}).copy()
            planner_ns["pivot_selection_failed"] = True
            planner_ns["needs_expansion"] = False
            md["planner"] = planner_ns
            return type(state)(**{**state, "metadata": md})

        # Update metadata with selected pivot and set phase to expansion
        md = metadata.copy()
        planner_ns = md.get("planner", {}).copy()
        planner_ns["pivot"] = pivot
        planner_ns["phase"] = "expansion"  # Transition to expansion phase
        md["planner"] = planner_ns

        logger.info(
            "[pivot_transition] Pivot validated and selected - transitioning to expansion phase"
        )
        logger.info(
            "[pivot_transition] Selected pivot | position=%.4f, score=%.4f, line=%d",
            pivot["position"], pivot["score"], pivot["line_number"]
        )
        logger.info("[pivot_transition] Pivot content: %s", _clip(pivot["content"], 200))

        return type(state)(
            messages=state.get("messages", []),
            context_messages=state.get("context_messages", []),
            metadata=md,
            current_agent=state.get("current_agent", ""),
            session_id=state.get("session_id", "")
        )

    return node_function
