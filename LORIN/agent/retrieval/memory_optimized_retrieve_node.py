"""
Memory-Optimized retrieve_node - Drop-in replacement for production CUDA memory safety
====================================================================================

Provides memory-safe retrieve_node() function that prevents CUDA out of memory errors
by using singleton FAISS engine management with aggressive memory cleanup.

Key Features:
- Drop-in replacement for original retrieve_node()
- Singleton FAISS engine prevents memory accumulation
- Real-time memory monitoring and cleanup
- Automatic recovery from memory errors
- Identical API and return format

Usage:
    Replace import in graph.py:

    # OLD:
    from .retrieve import retrieve_node

    # NEW:
    from .memory_optimized_retrieve_node import retrieve_node
"""

from typing import List, Dict, Any
from dataclasses import asdict
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage

from ..state import AgentState
from .memory_optimized_retriever import MemoryOptimizedRetriever
from .retrieve import (
    merge_retrieval_results, merge_search_stats, logger, SearchResult
)

# ─────────────────────────────────────────────────────────
# Temporal Filtering Utility
# ─────────────────────────────────────────────────────────

def apply_temporal_filter(
    results: List[SearchResult],
    temporal_constraint: str,
    global_region: str = "ALL"
) -> List[SearchResult]:
    """
    Filter search results based on temporal constraint

    Args:
        results: List of SearchResult objects
        temporal_constraint: String like "EARLY (0-30%)", "MIDDLE", "LATE", "ALL"
        global_region: Global temporal region from planner ("EARLY", "MIDDLE", "LATE", "ALL")

    Returns:
        Filtered list of SearchResult objects
    """
    if not temporal_constraint or "ALL" in temporal_constraint.upper():
        return results

    # Parse temporal constraint to get region bounds
    constraint_upper = temporal_constraint.upper()

    if "EARLY" in constraint_upper:
        region_start, region_end = 0.0, 0.3
    elif "MIDDLE" in constraint_upper:
        region_start, region_end = 0.3, 0.7
    elif "LATE" in constraint_upper:
        region_start, region_end = 0.7, 1.0
    else:
        # If constraint doesn't match known regions, don't filter
        logger.warning(f"[apply_temporal_filter] Unknown temporal constraint: {temporal_constraint}, skipping filter")
        return results

    logger.info(f"[apply_temporal_filter] Filtering with constraint '{temporal_constraint}' → region [{region_start}, {region_end}]")

    # Filter results based on relative_position in metadata
    filtered = []
    for result in results:
        # Get relative position from metadata
        rel_pos = result.metadata.get("relative_position")

        if rel_pos is None:
            # Fallback: estimate from line_number if relative_position not available
            line_num = result.metadata.get("line_number", 0)
            total_lines = result.metadata.get("total_lines", 3000)  # Default estimate
            rel_pos = line_num / total_lines if total_lines > 0 else 0.5

            if line_num > 0:  # Only log if we actually have a line_number
                logger.debug(f"[apply_temporal_filter] Estimated rel_pos={rel_pos:.3f} from line {line_num}/{total_lines}")

        # Check if within temporal region
        if region_start <= rel_pos <= region_end:
            filtered.append(result)

    logger.info(f"[apply_temporal_filter] Filtered {len(results)} → {len(filtered)} results (kept {len(filtered)/len(results)*100:.1f}%)")

    return filtered

# ─────────────────────────────────────────────────────────
# Memory-Optimized retrieve_node Function
# ─────────────────────────────────────────────────────────

async def retrieve_node(state: AgentState) -> AgentState:
    """
    Memory-optimized LangGraph-compatible retrieval node with PARALLEL subquery processing

    **PARALLEL EXECUTION**: Processes multiple subqueries concurrently for improved performance.

    This function provides identical functionality to the original retrieve_node but with:
    - Superior memory management through singleton FAISS engine reuse
    - PARALLEL subquery processing with asyncio.gather
    - Real-time GPU memory monitoring and cleanup

    PREVENTS CUDA OOM BY:
    - Using singleton MemoryOptimizedRetriever instead of creating new instances
    - Real-time GPU memory monitoring and cleanup
    - Automatic memory pressure detection and recovery
    - Aggressive garbage collection and cache clearing

    Args:
        state: Current agent state from LangGraph

    Returns:
        AgentState: Updated state with retrieval results (identical format to original)
    """
    logger.info("[memory_optimized_retrieve_node] Starting PARALLEL FAISS retrieval processing")

    # Extract existing data (same as original)
    messages: List[BaseMessage] = state.get("messages", [])
    metadata: Dict[str, Any] = state.get("metadata", {})

    # ==========================================
    # 🔬 Ablation: wo_Planner (skip_planner) Support
    # ==========================================
    experiment_config = metadata.get("experiment_config", {})
    skip_planner = experiment_config.get("skip_planner", False)

    if skip_planner:
        # wo_Planner: Use original question directly as single subquery
        logger.info("[memory_optimized_retrieve_node] 🔬 Ablation: wo_Planner - Using original question")

        # Get original question from first user message
        user_message = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                user_message = msg.content
                break

        if not user_message:
            logger.error("[memory_optimized_retrieve_node] No user message found for wo_Planner ablation")
            return state

        # Create a single synthetic subquery
        subqueries = [{
            "id": "Q1",
            "text": user_message,
            "type": "original_question",
            "temporal_constraint": "ALL"
        }]
        logger.info(f"[memory_optimized_retrieve_node] wo_Planner: Created single query Q1: '{user_message[:100]}...'")

        # No temporal context in wo_Planner mode
        global_temporal_region = "ALL"
        temporal_ctx = {}
    else:
        # Normal flow: Extract planner results
        planner_data = metadata.get("planner", {})
        if not planner_data:
            logger.warning("[memory_optimized_retrieve_node] No planner data found in metadata")
            return state

        plan_json = planner_data.get("last_plan_json", {})
        subqueries = plan_json.get("subqueries", [])

        # ✨ Extract temporal_context from planner (new field from improved prompt)
        temporal_ctx = planner_data.get("temporal_context", {})
        global_temporal_region = temporal_ctx.get("region", "ALL")

        if temporal_ctx:
            logger.info(
                "[memory_optimized_retrieve_node] Temporal context detected | region=%s, reasoning=%s",
                global_temporal_region,
                temporal_ctx.get("reasoning", "N/A")[:100]
            )

    if not subqueries:
        logger.warning("[memory_optimized_retrieve_node] No subqueries found in planner data")
        return state

    # Check for partial search mode (with experiment config support)
    replanner_data = metadata.get("replanner", {})

    # 🔧 중복 실행 방지 - LangGraph 상태 전달 타이밍 문제 해결
    if replanner_data.get("status") == "queries_consumed":
        logger.warning("[memory_optimized_retrieve_node] Queries already consumed, skipping retrieval to prevent infinite loop")
        return state

    # Partial search 조건: Replanner가 완료되었을 때
    is_partial_search = replanner_data.get("status") == "completed"

    if is_partial_search:
        # Partial search mode: only process reconstructed subqueries
        reconstructions = replanner_data.get("reconstructions", [])
        reconstructed_query_ids = [recon["subquery_id"] for recon in reconstructions]

        if not reconstructed_query_ids:
            logger.warning("[memory_optimized_retrieve_node] Partial search mode but no reconstructed queries found")
            return state

        logger.info(f"[memory_optimized_retrieve_node] Partial search mode: Processing {len(reconstructed_query_ids)} reconstructed subqueries")
        search_mode = "partial"
    else:
        # Full search mode: process all subqueries
        logger.info(f"[memory_optimized_retrieve_node] Full search mode: Processing {len(subqueries)} subqueries")
        search_mode = "full"

    # Initialize memory-optimized retriever (SINGLETON - no new instances!)
    try:
        # Create retrieval config with default settings
        from .retrieve import RetrievalConfig
        retrieval_config = RetrievalConfig()
        retriever = MemoryOptimizedRetriever(config=retrieval_config)

        # Log memory status before retrieval
        memory_stats = retriever.get_memory_statistics()
        logger.info(f"[memory_optimized_retrieve_node] Pre-retrieval memory: GPU {memory_stats.get('gpu_utilization_pct', 0):.1f}% used")

        # 🔬 FAISS 호출 횟수 카운터 초기화
        current_faiss_calls = 0

        # 🔧 LLM provider 정보 추출 (EXAONE 감지용)
        llm_provider = metadata.get("llm_provider", "").lower()

        if search_mode == "partial":
            # Execute adaptive partial retrieval for reconstructed queries only
            results_by_qid, search_stats = await retriever.search_specific_subqueries(
                all_subqueries=subqueries,
                target_query_ids=reconstructed_query_ids,
                provider=llm_provider
            )
            # Partial search: 재구성된 서브쿼리 개수만큼 FAISS 호출
            current_faiss_calls = len(reconstructed_query_ids)
        else:
            # Execute adaptive full batch retrieval
            results_by_qid, search_stats = await retriever.search_subqueries(subqueries, provider=llm_provider)
            # Full search: 모든 서브쿼리 개수만큼 FAISS 호출
            current_faiss_calls = len(subqueries)

        # 🔬 FAISS 호출 횟수 누적 (실험용 메트릭)
        total_faiss_calls = metadata.get("faiss_calls", 0) + current_faiss_calls
        logger.info(f"[memory_optimized_retrieve_node] FAISS calls: {current_faiss_calls} this round, {total_faiss_calls} total")

        # Log memory status after retrieval
        memory_stats = retriever.get_memory_statistics()
        logger.info(f"[memory_optimized_retrieve_node] Post-retrieval memory: GPU {memory_stats.get('gpu_utilization_pct', 0):.1f}% used")

        # Convert results to serializable format (same as original)
        new_serializable_results = {}
        for qid, result in results_by_qid.items():
            logger.info(f"[memory_optimized_retrieve_node] Converting QID {qid}: {len(result.results)} results")
            converted_results = []
            for i, r in enumerate(result.results):
                logger.info(f"[memory_optimized_retrieve_node] QID {qid}, result {i}: type={type(r)}")
                try:
                    converted = asdict(r)
                    logger.info(f"[memory_optimized_retrieve_node] QID {qid}, result {i}: asdict success, type={type(converted)}")
                    converted_results.append(converted)
                except Exception as e:
                    logger.error(f"[memory_optimized_retrieve_node] QID {qid}, result {i}: asdict error: {e}")
                    # Fallback: if it's already a dict, use it directly
                    if isinstance(r, dict):
                        converted_results.append(r)
                    else:
                        logger.error(f"[memory_optimized_retrieve_node] QID {qid}, result {i}: cannot convert {type(r)}")

            new_serializable_results[qid] = {
                "query_id": result.query_id,
                "query_text": result.query_text,
                "results": converted_results,
                "search_time": result.search_time,
                "result_count": result.result_count,
                "avg_score": result.avg_score,
                "max_score": result.max_score,
                "min_score": result.min_score,
                "cache_hit": result.cache_hit,
                "error": result.error
            }

        # Handle result merging for partial search (same as original)
        if search_mode == "partial":
            # Preserve existing results and merge with new ones
            existing_retriever_data = metadata.get("faiss_retriever", {})
            existing_results = existing_retriever_data.get("results_by_qid", {})

            # Debug existing results structure
            logger.info(f"[memory_optimized_retrieve_node] Existing results: {len(existing_results)} QIDs")
            for qid, result_data in existing_results.items():
                logger.info(f"[memory_optimized_retrieve_node] Existing QID {qid}: type={type(result_data)}")
                if isinstance(result_data, dict) and "results" in result_data:
                    results_list = result_data["results"]
                    logger.info(f"[memory_optimized_retrieve_node] Existing QID {qid} results: {len(results_list)} items")
                    if results_list:
                        logger.info(f"[memory_optimized_retrieve_node] Existing QID {qid}, first item type: {type(results_list[0])}")
                else:
                    logger.warning(f"[memory_optimized_retrieve_node] Existing QID {qid}: unexpected structure")
            existing_stats = existing_retriever_data.get("search_stats", {})

            # Merge results
            final_serializable_results = merge_retrieval_results(
                existing_results, new_serializable_results, search_stats
            )

            # Merge statistics
            final_search_stats = merge_search_stats(existing_stats, search_stats, "incremental")

            summary_content = (
                f"FAISS Partial Retrieval Complete (Memory-Optimized)\n"
                f"Reconstructed: {len(reconstructed_query_ids)} subqueries\n"
                f"New Results: {search_stats.total_results} items\n"
                f"Total Results: {final_search_stats.get('total_results', 0)} items\n"
                f"Average Score: {search_stats.avg_score:.4f}\n"
                f"Search Time: {search_stats.search_time:.2f}s\n"
                f"GPU Memory: {memory_stats.get('gpu_utilization_pct', 0):.1f}% used"
            )
        else:
            # Full search: use results as-is
            final_serializable_results = new_serializable_results
            final_search_stats = asdict(search_stats)

            summary_content = (
                f"FAISS Full Retrieval Complete (Memory-Optimized)\n"
                f"Processed: {len(subqueries)} subqueries\n"
                f"Found: {search_stats.total_results} total results\n"
                f"Average Score: {search_stats.avg_score:.4f}\n"
                f"Cache Hits: {search_stats.cache_hits}/{len(subqueries)}\n"
                f"Search Time: {search_stats.search_time:.2f}s\n"
                f"GPU Memory: {memory_stats.get('gpu_utilization_pct', 0):.1f}% used"
            )

        # Create queries mapping from final results (same as original)
        queries_by_qid = {}
        for qid, result_data in final_serializable_results.items():
            queries_by_qid[qid] = result_data.get("query_text", "")

        # Store in metadata.faiss_retriever namespace (same as original)
        retriever_metadata = {
            "results_by_qid": final_serializable_results,
            "queries_by_qid": queries_by_qid,
            "search_stats": final_search_stats,
            "config": {
                "index_path": retriever.config.INDEX_PATH,
                "top_k": retriever.config.TOP_K_DEFAULT,
                "adaptive_k": retriever.config.ADAPTIVE_K_ENABLED,
                "deduplication": retriever.config.DEDUPLICATION_ENABLED
            },
            "search_mode": search_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "memory_optimized": True,  # Flag to indicate this used memory optimization
            "memory_stats": memory_stats  # Include memory statistics
        }

        # Update metadata safely (same as original + experiment metrics)
        new_metadata = metadata.copy()
        new_metadata["faiss_retriever"] = retriever_metadata
        new_metadata["last_agent"] = "faiss_retriever"
        new_metadata["faiss_calls"] = total_faiss_calls  # 🔬 실험용 메트릭 저장

        if search_mode == "partial":
            # Mark reconstructed queries as consumed so routing doesn't retrigger retrieve
            replanner_meta = (new_metadata.get("replanner") or {}).copy()
            if replanner_meta:
                if replanner_meta.get("status") == "completed":
                    replanner_meta["status"] = "queries_consumed"
                if replanner_meta.get("reconstructions"):
                    # 🔧 reconstructions를 비우지 않고 사용됨을 표시
                    for recon in replanner_meta["reconstructions"]:
                        recon["consumed"] = True
                        recon["consumed_at"] = datetime.now(timezone.utc).isoformat()
                replanner_meta["last_consumed_at"] = datetime.now(timezone.utc).isoformat()
                new_metadata["replanner"] = replanner_meta

        retrieval_message = AIMessage(
            content=summary_content,
            additional_kwargs={
                "agent_name": "faiss_retriever",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "search_mode": search_mode,
                "retrieval_stats": final_search_stats if search_mode == "full" else asdict(search_stats),
                "memory_optimized": True,
                "memory_stats": memory_stats
            }
        )

        new_messages = messages + [retrieval_message]

        logger.info(f"[memory_optimized_retrieve_node] Successfully completed {search_mode} search")
        if search_mode == "partial":
            logger.info(f"[memory_optimized_retrieve_node] Partial results: {search_stats.total_results} new, {final_search_stats.get('total_results', 0)} total")
        else:
            logger.info(f"[memory_optimized_retrieve_node] Full results: {search_stats.total_results}, Avg score: {search_stats.avg_score:.4f}")

        # Return updated state (same format as original)
        return type(state)(
            messages=new_messages,
            context_messages=state.get("context_messages", []),
            metadata=new_metadata,
            current_agent="faiss_retriever",
            session_id=state.get("session_id", "")
        )

    except Exception as e:
        error_msg = f"Memory-optimized FAISS retrieval failed: {str(e)}"
        logger.error(f"[memory_optimized_retrieve_node] {error_msg}")

        # Enhanced error handling with memory stats
        try:
            # Try to get memory stats even on error
            if 'retriever' in locals():
                memory_stats = retriever.get_memory_statistics()
                logger.error(f"[memory_optimized_retrieve_node] Error memory state: {memory_stats}")
        except:
            logger.error("[memory_optimized_retrieve_node] Could not retrieve memory stats on error")

        # Store error in metadata (same as original but with memory info)
        new_metadata = metadata.copy()
        new_metadata["faiss_retriever"] = {
            "success": False,
            "error": error_msg,
            "search_mode": search_mode if 'search_mode' in locals() else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_optimized": True,
            "memory_stats": memory_stats if 'memory_stats' in locals() else {}
        }
        new_metadata["last_agent"] = "faiss_retriever"

        # Create error message
        error_message = AIMessage(
            content=f"Memory-Optimized Retrieval Error: {error_msg}",
            additional_kwargs={
                "agent_name": "faiss_retriever",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": True,
                "memory_optimized": True
            }
        )

        new_messages = messages + [error_message]

        return type(state)(
            messages=new_messages,
            context_messages=state.get("context_messages", []),
            metadata=new_metadata,
            current_agent="faiss_retriever",
            session_id=state.get("session_id", "")
        )

# ─────────────────────────────────────────────────────────
# Utility Functions (same as original but with memory stats)
# ─────────────────────────────────────────────────────────

def get_memory_optimized_retrieval_stats() -> Dict[str, Any]:
    """Get memory and performance statistics from the singleton retriever"""
    try:
        from .memory_optimized_retriever import MemoryOptimizedRetriever
        retriever = MemoryOptimizedRetriever()
        return retriever.get_memory_statistics()
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        return {}

def force_memory_cleanup():
    """Force aggressive memory cleanup on the singleton manager"""
    try:
        from .memory_manager import MemoryOptimizedFAISSManager
        manager = MemoryOptimizedFAISSManager()
        manager._emergency_cleanup()
        logger.info("Forced memory cleanup completed")
        return True
    except Exception as e:
        logger.error(f"Force cleanup failed: {e}")
        return False

# ─────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────

__all__ = [
    "retrieve_node",
    "get_memory_optimized_retrieval_stats",
    "force_memory_cleanup"
]
