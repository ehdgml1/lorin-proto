"""
LORIN System Metadata Schema Management

Module for metadata schema definitions and safe access patterns.
Provides type-safe metadata structures for inter-node communication.

Features:
- TypedDict-based metadata schema definitions
- Safe metadata access and validation
- Inter-node compatibility support (key mapping)
- Metadata creation and update utilities
"""

from __future__ import annotations
from typing import TypedDict, Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# --- Common Data Structure TypedDict Definitions ---

class SubqueryInfo(TypedDict, total=False):
    """Subquery information schema"""
    query_id: str                 # Query ID (e.g., "Q1", "Q2")
    query_text: str               # Query text
    original_query: str           # Original query text
    depends_on: Optional[str]     # Dependency query ID
    is_anchor: bool               # Anchor query flag
    keywords: List[str]           # Extracted keywords
    priority: int                 # Priority


class FailedQueryInfo(TypedDict, total=False):
    """Failed query information schema"""
    query_id: str                 # Query ID
    query_text: str               # Query text
    relevance_score: float        # Relevance score
    confidence: float             # Confidence
    reasoning: str                # Failure reason


class QueryEvaluationInfo(TypedDict, total=False):
    """Query evaluation result schema"""
    query_id: str                 # Query ID
    query_text: str               # Query text
    is_relevant: bool             # Relevance flag
    relevance_score: float        # Relevance score (0.0 ~ 1.0)
    confidence: float             # Evaluation confidence (0.0 ~ 1.0)
    reasoning: str                # Evaluation reasoning
    document_count: int           # Number of evaluated documents


class PatternDetectionResult(TypedDict):
    """Pattern detection result schema"""
    detected: bool                # Detection flag
    pattern: Optional[str]        # Detected pattern (e.g., "replanner-retrieve")
    confidence: float             # Detection confidence


class LoopDetectionResult(TypedDict):
    """Loop detection result schema"""
    is_loop: bool                 # Loop detection flag
    pattern: Optional[str]        # Detected pattern description
    confidence: float             # Detection confidence


class LoopPreventionData(TypedDict, total=False):
    """Loop prevention system data schema"""
    global_iteration_count: int   # Global iteration count
    routing_history: List[str]    # Routing history (e.g., ["replanner", "retrieve", ...])
    last_routes: List[str]        # Recent routing list
    pattern_cache: Dict[str, Any] # Pattern cache
    start_timestamp: str          # Start time (ISO format)
    forced_terminations: int      # Forced termination count
    oscillation_count: int        # Oscillation detection count


class RoutingDecision(TypedDict, total=False):
    """Routing decision result schema"""
    route: str                    # Routing target ("replanner", "retrieve", "answer")
    reason: str                   # Decision reason
    strategy: Optional[str]       # Applied strategy


class LastRoutingDecision(TypedDict, total=False):
    """Last routing decision information schema"""
    route: str                    # Selected routing
    reason: str                   # Decision reason
    strategy: str                 # Applied strategy
    timestamp: str                # Decision time (ISO format)
    global_iteration: int         # Global iteration count


class IterationHistoryEntry(TypedDict, total=False):
    """Iteration history entry schema"""
    iteration: int                # Iteration number
    failed_count: int             # Failed query count
    success_count: int            # Successful query count
    strategy: str                 # Applied strategy
    timestamp: str                # Timestamp


class QuerySuccessTimelineEntry(TypedDict, total=False):
    """Query success timeline entry schema"""
    iteration: int                # Iteration number at success
    success_timestamp: str        # Success time
    search_timestamp: str         # Search time
    relevance_score: float        # Relevance score
    confidence: float             # Confidence
    first_success: bool           # First success flag
    latest_iteration: Optional[int]          # Latest success iteration (on re-evaluation)
    latest_success_timestamp: Optional[str]  # Latest success time
    latest_search_timestamp: Optional[str]   # Latest search time
    latest_relevance_score: Optional[float]  # Latest relevance score
    latest_confidence: Optional[float]       # Latest confidence


class LoopPreventionStatus(TypedDict, total=False):
    """Loop prevention status schema (for answer node)"""
    terminated: bool              # Forced termination flag
    reason: str                   # Termination reason
    global_iterations: int        # Global iteration count
    forced_terminations: int      # Forced termination count
    routing_history: List[str]    # Recent routing history
    strategy: str                 # Applied strategy
    has_partial_results: bool     # Partial results available flag


# --- Per-Node Metadata Schema Definitions ---

class PlannerMetadata(TypedDict, total=False):
    """Planner node metadata schema"""
    last_plan_json: Dict[str, Any]  # Main plan JSON
    n_subqueries: int               # Number of subqueries
    depends_mode: str               # Dependency mode ("chain", "parallel", etc.)
    has_anchor: bool                # Anchor query existence flag
    error: Optional[str]            # Error message (optional)
    timestamp: str                  # Creation time

class RetrieveMetadata(TypedDict, total=False):
    """Retrieve node metadata schema"""
    results_by_qid: Dict[str, List[Dict[str, Any]]]  # Search results by QID
    queries_by_qid: Dict[str, str]                   # Query text by QID
    search_stats: Dict[str, Any]                     # Search statistics
    config: Dict[str, Any]                           # Configuration
    search_mode: str                                 # "full" or "partial"
    timestamp: str                                   # Execution time
    success: bool                                    # Success flag

class QualityEvaluatorMetadata(TypedDict, total=False):
    """Quality Evaluator node metadata schema"""
    evaluation_results: Dict[str, Any]           # Current iteration results
    cumulative_evaluation_results: Dict[str, Any]  # Cumulative results (success preserved)
    relevance_summary: Dict[str, Any]           # Current iteration summary
    cumulative_relevance_summary: Dict[str, Any] # Cumulative summary
    evaluation_stats: Dict[str, Any]            # Evaluation statistics
    config: Dict[str, Any]                      # Configuration
    evaluation_mode: str                        # "initial" or "cumulative"
    iteration_count: int                        # Iteration count
    timestamp: str                              # Evaluation time
    success: bool                               # Success flag

class ReplannerMetadata(TypedDict, total=False):
    """Replanner node metadata schema"""
    status: str                     # "completed", "converged", "error"
    iteration_count: int            # Iteration count
    failed_count: int              # Failed query count
    reconstructed_count: int       # Reconstructed query count
    failure_analyses: List[Dict[str, Any]]  # Failure analysis results
    reconstructions: List[Dict[str, Any]]   # Reconstruction results
    convergence_reason: Optional[str]       # Convergence reason (optional)
    iteration_history: List[Dict[str, Any]] # Iteration history
    error_message: Optional[str]            # Error message (optional)
    timestamp: str                          # Replan time

class AnswerMetadata(TypedDict, total=False):
    """Answer node metadata schema"""
    evidence_blocks: Dict[str, Any]      # Full evidence label map
    used_queries_by_qid: Dict[str, str]  # QID to query mapping
    subanswers_by_qid: Dict[str, str]    # Sub-answer by QID
    evidence_by_qid: Dict[str, Dict]     # Evidence label map by QID
    final: Dict[str, str]                # Final answer info
    timestamp: str                       # Answer generation time

class LorinMetadata(TypedDict, total=False):
    """Full LORIN system metadata schema"""
    planner: PlannerMetadata
    faiss_retriever: RetrieveMetadata    # Standard key
    retriever: RetrieveMetadata          # Compatibility key (deprecated)
    quality_evaluator: QualityEvaluatorMetadata
    replanner: ReplannerMetadata
    answer: AnswerMetadata


# --- Metadata Manager ---

@dataclass
class MetadataValidationResult:
    """Metadata validation result"""
    is_valid: bool
    missing_fields: List[str]
    errors: List[str]
    warnings: List[str]

class MetadataManager:
    """LORIN system metadata manager

    Handles safe metadata access, validation, and creation.
    """

    # Compatibility key mapping (deprecated -> standard)
    COMPATIBILITY_MAPPING = {
        "retriever": "faiss_retriever"
    }

    # Required fields per node
    REQUIRED_FIELDS = {
        "planner": ["last_plan_json", "n_subqueries"],
        "faiss_retriever": ["results_by_qid", "success"],
        "quality_evaluator": ["evaluation_results", "success"],
        "replanner": ["status", "iteration_count"],
        "answer": ["final"]
    }

    @staticmethod
    def safe_get_metadata(state: Dict[str, Any], node_key: str, field_key: Optional[str] = None) -> Any:
        """Safe metadata access

        Args:
            state: AgentState dictionary
            node_key: Node key (e.g., "planner", "faiss_retriever")
            field_key: Field key (optional)

        Returns:
            Metadata value or None
        """
        try:
            # Check compatibility mapping
            actual_key = MetadataManager.COMPATIBILITY_MAPPING.get(node_key, node_key)

            metadata = state.get("metadata", {})

            # Try standard key
            node_data = metadata.get(actual_key)

            # Compatibility key fallback
            if node_data is None and actual_key != node_key:
                node_data = metadata.get(node_key)

            if node_data is None:
                logger.debug(f"Node metadata not found: {node_key} (tried: {actual_key})")
                return None

            # Return specific field if requested
            if field_key:
                return node_data.get(field_key)

            return node_data

        except Exception as e:
            logger.error(f"Error accessing metadata[{node_key}][{field_key}]: {e}")
            return None

    @staticmethod
    def safe_set_metadata(state: Dict[str, Any], node_key: str, data: Dict[str, Any],
                         add_timestamp: bool = True) -> bool:
        """Safe metadata setting

        Args:
            state: AgentState dictionary
            node_key: Node key
            data: Data to set
            add_timestamp: Auto-add timestamp flag

        Returns:
            Success flag
        """
        try:
            if "metadata" not in state:
                state["metadata"] = {}

            # Add timestamp
            if add_timestamp and "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc).isoformat()

            state["metadata"][node_key] = data

            # Compatibility support: copy faiss_retriever to retriever
            if node_key == "faiss_retriever":
                state["metadata"]["retriever"] = data.copy()

            logger.debug(f"Set metadata for {node_key}: {len(data)} fields")
            return True

        except Exception as e:
            logger.error(f"Error setting metadata[{node_key}]: {e}")
            return False

    @staticmethod
    def validate_node_metadata(state: Dict[str, Any], node_key: str) -> MetadataValidationResult:
        """Validate node metadata

        Args:
            state: AgentState dictionary
            node_key: Node key to validate

        Returns:
            Validation result
        """
        result = MetadataValidationResult(
            is_valid=True,
            missing_fields=[],
            errors=[],
            warnings=[]
        )

        try:
            # Check metadata existence
            node_data = MetadataManager.safe_get_metadata(state, node_key)
            if node_data is None:
                result.is_valid = False
                result.errors.append(f"Node metadata '{node_key}' not found")
                return result

            # Check required fields
            required_fields = MetadataManager.REQUIRED_FIELDS.get(node_key, [])
            for field in required_fields:
                if field not in node_data or node_data[field] is None:
                    result.missing_fields.append(field)
                    result.is_valid = False

            # Check success field (if exists)
            if "success" in node_data and not node_data["success"]:
                result.warnings.append(f"Node '{node_key}' marked as failed")

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {e}")

        return result

    @staticmethod
    def get_metadata_summary(state: Dict[str, Any]) -> Dict[str, str]:
        """Get full metadata summary

        Returns:
            Status summary by node
        """
        summary = {}
        metadata = state.get("metadata", {})

        for node_key in ["planner", "faiss_retriever", "quality_evaluator", "replanner", "answer"]:
            node_data = metadata.get(node_key)
            if node_data:
                success = node_data.get("success", "unknown")
                timestamp = node_data.get("timestamp", "none")
                summary[node_key] = f"status={success}, time={timestamp[:19] if timestamp != 'none' else 'none'}"
            else:
                summary[node_key] = "not_found"

        return summary


# --- Per-Node Helper Functions ---

def create_planner_metadata(last_plan_json: Dict[str, Any], n_subqueries: int,
                           depends_mode: str = "chain", has_anchor: bool = True,
                           error: Optional[str] = None) -> PlannerMetadata:
    """Create Planner metadata"""
    return PlannerMetadata(
        last_plan_json=last_plan_json,
        n_subqueries=n_subqueries,
        depends_mode=depends_mode,
        has_anchor=has_anchor,
        error=error,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

def create_retrieve_metadata(results_by_qid: Dict[str, List[Dict]],
                           queries_by_qid: Dict[str, str],
                           search_mode: str = "full",
                           success: bool = True) -> RetrieveMetadata:
    """Create Retrieve metadata"""
    return RetrieveMetadata(
        results_by_qid=results_by_qid,
        queries_by_qid=queries_by_qid,
        search_stats={
            "total_results": sum(len(results) for results in results_by_qid.values()),
            "queries_processed": len(queries_by_qid)
        },
        config={},
        search_mode=search_mode,
        success=success,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

def create_quality_evaluator_metadata(evaluation_results: Dict[str, Any],
                                     evaluation_mode: str = "initial",
                                     iteration_count: int = 1,
                                     success: bool = True) -> QualityEvaluatorMetadata:
    """Create Quality Evaluator metadata"""
    return QualityEvaluatorMetadata(
        evaluation_results=evaluation_results,
        cumulative_evaluation_results=evaluation_results.copy(),
        relevance_summary={},
        cumulative_relevance_summary={},
        evaluation_stats={},
        config={},
        evaluation_mode=evaluation_mode,
        iteration_count=iteration_count,
        success=success,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

def create_replanner_metadata(status: str, iteration_count: int,
                             failed_count: int = 0, reconstructed_count: int = 0,
                             error_message: Optional[str] = None) -> ReplannerMetadata:
    """Create Replanner metadata"""
    return ReplannerMetadata(
        status=status,
        iteration_count=iteration_count,
        failed_count=failed_count,
        reconstructed_count=reconstructed_count,
        failure_analyses=[],
        reconstructions=[],
        convergence_reason=None,
        iteration_history=[],
        error_message=error_message,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

def create_answer_metadata(final_output: str, used_queries: Dict[str, str]) -> AnswerMetadata:
    """Create Answer metadata"""
    return AnswerMetadata(
        evidence_blocks={},
        used_queries_by_qid=used_queries,
        subanswers_by_qid={},
        evidence_by_qid={},
        final={
            "prompt": "",
            "output": final_output,
            "raw": final_output
        },
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# --- Convenience Functions ---

def get_subqueries(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract subquery list generated by planner"""
    planner_data = MetadataManager.safe_get_metadata(state, "planner")
    if planner_data and "last_plan_json" in planner_data:
        return planner_data["last_plan_json"].get("subqueries", [])
    return []

def get_search_results(state: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Extract search results (with compatibility support)"""
    logger.info(f"[get_search_results] Extracting search results from state")

    # Try faiss_retriever first
    faiss_data = MetadataManager.safe_get_metadata(state, "faiss_retriever")
    logger.info(f"[get_search_results] faiss_data found: {faiss_data is not None}")

    if faiss_data and "results_by_qid" in faiss_data:
        results_by_qid = faiss_data["results_by_qid"]
        logger.info(f"[get_search_results] faiss_retriever results: {len(results_by_qid)} QIDs")

        # Convert from new format to expected format
        converted_results = {}
        for qid, qid_data in results_by_qid.items():
            logger.info(f"[get_search_results] Converting QID {qid}: type={type(qid_data)}")
            if isinstance(qid_data, dict) and "results" in qid_data:
                actual_results = qid_data["results"]
                logger.info(f"[get_search_results] QID {qid} has {len(actual_results)} results")
                if actual_results:
                    logger.info(f"[get_search_results] QID {qid}, first result type: {type(actual_results[0])}")
                converted_results[qid] = actual_results
            else:
                logger.warning(f"[get_search_results] QID {qid}: unexpected format, using as-is")
                converted_results[qid] = qid_data

        logger.info(f"[get_search_results] Converted results: {len(converted_results)} QIDs")
        return converted_results

    # retriever fallback
    retriever_data = MetadataManager.safe_get_metadata(state, "retriever")
    if retriever_data and "results_by_qid" in retriever_data:
        return retriever_data["results_by_qid"]

    return {}

def get_quality_filtered_search_results(state: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Return quality-filtered search results based on Quality Evaluator results

    This function applies the following logic:
    1. Queries that passed on first evaluation -> use results from that point
    2. Queries that passed after replanning -> use only latest results
    3. Intermediate results from failed queries are completely excluded

    Args:
        state: AgentState dictionary

    Returns:
        Dict[str, List[Dict]]: Quality-filtered search results
    """
    logger.info("[get_quality_filtered_search_results] Starting quality-based result filtering")

    # 1. Check Quality Evaluator results
    evaluator_data = MetadataManager.safe_get_metadata(state, "quality_evaluator")
    if not evaluator_data or not evaluator_data.get("success", False):
        logger.warning("[get_quality_filtered_search_results] No quality evaluator results found, falling back to all results")
        return get_search_results(state)

    # 2. Extract query success timeline data
    query_success_timeline = evaluator_data.get("query_success_timeline", {})
    if not query_success_timeline:
        logger.warning("[get_quality_filtered_search_results] No success timeline found, falling back to all results")
        return get_search_results(state)

    # 3. Get original search results
    all_search_results = get_search_results(state)
    if not all_search_results:
        logger.warning("[get_quality_filtered_search_results] No search results found")
        return {}

    # 4. Filter only successful queries
    filtered_results = {}
    success_count = 0
    first_success_count = 0
    latest_success_count = 0

    for qid, success_info in query_success_timeline.items():
        if qid not in all_search_results:
            logger.warning(f"[get_quality_filtered_search_results] Success timeline has {qid} but no search results found")
            continue

        # Include search results for successful queries
        filtered_results[qid] = all_search_results[qid]
        success_count += 1

        # Classify success type
        if success_info.get("first_success", False):
            if "latest_iteration" not in success_info:
                # First success without re-evaluation
                first_success_count += 1
                logger.info(f"[get_quality_filtered_search_results] QID {qid}: First success at iteration {success_info.get('iteration')}")
            else:
                # First success with re-evaluation
                latest_success_count += 1
                logger.info(f"[get_quality_filtered_search_results] QID {qid}: First success at iteration {success_info.get('iteration')}, latest at {success_info.get('latest_iteration')}")

    # 5. Log filtering statistics
    total_original = len(all_search_results)
    total_filtered = len(filtered_results)
    excluded_count = total_original - total_filtered

    logger.info(f"[get_quality_filtered_search_results] Filtering complete:")
    logger.info(f"  - Original results: {total_original} queries")
    logger.info(f"  - Filtered results: {total_filtered} queries ({total_filtered/total_original*100:.1f}%)")
    logger.info(f"  - Excluded results: {excluded_count} queries")
    logger.info(f"  - First successes: {first_success_count} queries")
    logger.info(f"  - Latest successes: {latest_success_count} queries")

    # 6. Log excluded queries (for debugging)
    excluded_qids = set(all_search_results.keys()) - set(filtered_results.keys())
    if excluded_qids:
        logger.info(f"[get_quality_filtered_search_results] Excluded query IDs: {sorted(excluded_qids)}")

    return filtered_results

def is_node_completed(state: Dict[str, Any], node_key: str) -> bool:
    """Check if node execution is completed"""
    node_data = MetadataManager.safe_get_metadata(state, node_key)
    return node_data is not None and node_data.get("success", False)

def log_metadata_status(state: Dict[str, Any], context: str = ""):
    """Log metadata status"""
    summary = MetadataManager.get_metadata_summary(state)
    logger.info(f"Metadata status {context}: {summary}")
