"""
FAISS Retriever Module - Production-grade FAISS-based Information Retrieval
===========================================================================

Integrates with the LORIN system to provide efficient semantic search and retrieval
capabilities using FAISS vector index and BGE-multilingual-gemma2 embeddings.

Core Features:
- Subquery batch processing from planner output
- Adaptive k-value calculation based on query characteristics
- Semantic and text-based deduplication
- Advanced search statistics and performance monitoring
- Caching system for improved performance
- Safe state management preserving existing metadata

Architecture:
- FAISSRetriever: Main retrieval engine with advanced query processing
- SubqueryResult: Structured result container with metadata
- SearchStats: Performance and quality metrics tracking
- retrieve_node: LangGraph-compatible state processor

Integration:
- Input: Subqueries from metadata.planner
- Output: Search results in metadata.faiss_retriever namespace
- Dependencies: FAISSSearchEngine from search_engine.py
"""

from __future__ import annotations
import os
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import defaultdict, OrderedDict

from langchain_core.messages import AIMessage, BaseMessage

from ...make_faiss.search_engine import FAISSSearchEngine, SearchResult
from ...logger.logger import get_logger
from ..state import AgentState
from ..schema import MetadataManager

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# LRU Cache with TTL Implementation
# ─────────────────────────────────────────────────────────

class LRUCacheWithTTL:
    """LRU 캐시 + TTL 지원 - 메모리 효율적 캐싱"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회 (TTL 체크 포함)"""
        if key not in self.cache:
            return None

        # TTL 체크
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            self._evict(key)
            return None

        # LRU: 최근 사용으로 이동
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """캐시에 값 저장 (LRU + TTL)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.cache))
                self._evict(oldest_key)

            self.cache[key] = value

        self.timestamps[key] = time.time()

    def _evict(self, key: str) -> None:
        """캐시에서 항목 제거"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]

    def clear(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        self.timestamps.clear()

    def size(self) -> int:
        """현재 캐시 크기"""
        return len(self.cache)

# ─────────────────────────────────────────────────────────
# Data Classes for Structured Results
# ─────────────────────────────────────────────────────────

@dataclass
class SubqueryResult:
    """Individual subquery search result container"""
    query_id: str
    query_text: str
    results: List[SearchResult]
    search_time: float
    result_count: int
    avg_score: float
    max_score: float
    min_score: float
    cache_hit: bool = False
    error: Optional[str] = None

@dataclass
class SearchStats:
    """Aggregated search statistics and performance metrics"""
    total_queries: int
    total_results: int
    avg_score: float
    search_time: float
    cache_hits: int
    cache_hit_rate: float
    query_distribution: Dict[str, int]  # query types
    score_distribution: Dict[str, int]  # score ranges
    performance_metrics: Dict[str, float]

# ─────────────────────────────────────────────────────────
# Environment Configuration
# ─────────────────────────────────────────────────────────

def _get_env_int(name: str, default: int) -> int:
    """Get integer environment variable with fallback"""
    try:
        value = os.getenv(name)
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def _get_env_float(name: str, default: float) -> float:
    """Get float environment variable with fallback"""
    try:
        value = os.getenv(name)
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def _get_env_bool(name: str, default: bool) -> bool:
    """Get boolean environment variable with fallback"""
    value = (os.getenv(name, "").strip().lower())
    if not value:
        return default
    return value in ("1", "true", "yes", "y", "on")

def _get_env_str(name: str, default: str) -> str:
    """Get string environment variable with fallback"""
    return os.getenv(name, default).strip()

# ─────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────

class RetrievalConfig:
    """Centralized configuration for retrieval system"""

    def __init__(self, max_docs: Optional[int] = None):
        """Initialize retrieval config with optional max_docs override

        Args:
            max_docs: Override for TOP_K_DEFAULT (from experiment config)
        """
        # FAISS Index Configuration
        self.INDEX_PATH = _get_env_str("FAISS_INDEX_PATH", "LORIN/make_faiss/log_faiss_index_bge_gemma2")
        self.MODEL_NAME = _get_env_str("BGE_MODEL_NAME", "BAAI/bge-multilingual-gemma2")
        self.DEVICE = _get_env_str("BGE_DEVICE", "cuda:1")  # GPU 1에 임베딩 모델 로드 (EXAONE은 GPU 0)

        # Search Parameters
        # ✅ max_docs from experiment config overrides environment variable
        self.TOP_K_DEFAULT = max_docs if max_docs is not None else _get_env_int("RETRIEVER_TOP_K", 3)
        self.TOP_K_MAX = _get_env_int("RETRIEVER_TOP_K_MAX", 30)  # Phase 2: Increased max to allow adaptive scaling
        self.QUERY_MAX_LENGTH = _get_env_int("RETRIEVER_QUERY_MAXLEN", 200)

        # Adaptive Search Configuration
        self.ADAPTIVE_K_ENABLED = _get_env_bool("RETR_ADAPTIVE_INIT", True)
        self.SHORT_QUERY_WORDS = _get_env_int("RETR_SHORT_WORDS", 3)
        self.LONG_QUERY_WORDS = _get_env_int("RETR_LONG_WORDS", 12)

        # Quality Thresholds
        self.SUCCESS_THRESHOLD = _get_env_float("RETRIEVER_SUCCESS_THRESHOLD", 0.5)
        self.QUALITY_THRESHOLD = _get_env_float("RETRIEVER_QUALITY_THRESHOLD", 0.25)

        # Performance Settings
        self.SEARCH_TIMEOUT = _get_env_float("RETRIEVER_TIMEOUT", 30.0)
        self.CACHE_MAX_SIZE = _get_env_int("RETRIEVER_CACHE_SIZE", 1000)
        self.DEDUPLICATION_ENABLED = _get_env_bool("RETRIEVER_DEDUP", True)
        self.PARALLEL_ENABLED = _get_env_bool("RETRIEVER_PARALLEL", True)

# ─────────────────────────────────────────────────────────
# FAISS Retriever Implementation
# ─────────────────────────────────────────────────────────

class FAISSRetriever:
    """Production-grade FAISS-based retrieval engine with advanced features"""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize FAISS retriever with configuration

        Args:
            config: Optional configuration object, uses defaults if None
        """
        self.config = config or RetrievalConfig()
        self.search_engine: Optional[FAISSSearchEngine] = None
        # 최적화: 단순 dict → LRU + TTL 캐시
        self.query_cache = LRUCacheWithTTL(
            max_size=self.config.CACHE_MAX_SIZE,
            ttl_seconds=3600  # 1시간
        )
        self.stats_cache: Dict[str, Any] = {}
        self.initialized = False

        # Performance tracking
        self.total_searches = 0
        self.total_cache_hits = 0
        self.search_times: List[float] = []

        logger.info(f"[FAISSRetriever] Initializing with index: {self.config.INDEX_PATH}")
        logger.info(f"[FAISSRetriever] LRU Cache enabled: max_size={self.config.CACHE_MAX_SIZE}, ttl=3600s")

    def _initialize_search_engine(self) -> bool:
        """Initialize the FAISS search engine with error handling

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.initialized and self.search_engine is not None:
            return True

        try:
            logger.info(f"[FAISSRetriever] Loading search engine from {self.config.INDEX_PATH}")
            self.search_engine = FAISSSearchEngine(
                index_path=self.config.INDEX_PATH,
                model_name=self.config.MODEL_NAME,
                device=self.config.DEVICE,
                use_fp16=True,
                max_length=self.config.QUERY_MAX_LENGTH
            )
            self.initialized = True
            logger.info("[FAISSRetriever] Search engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"[FAISSRetriever] Failed to initialize search engine: {e}")
            self.initialized = False
            return False

    def _adaptive_k_calculation(self, query: str, base_k: int) -> int:
        """Calculate adaptive k value based on query characteristics

        Args:
            query: Search query text
            base_k: Base k value to adjust

        Returns:
            int: Adjusted k value optimized for query type
        """
        if not self.config.ADAPTIVE_K_ENABLED:
            return base_k

        words = query.split()
        word_count = len(words)

        # Fixed k value - no adaptive adjustment based on query length or keywords
        k_multiplier = 1.0

        adjusted_k = int(base_k * k_multiplier)
        return min(max(adjusted_k, 1), self.config.TOP_K_MAX)

    def _calculate_query_hash(self, query: str, k: int) -> str:
        """Generate cache key for query

        Args:
            query: Search query
            k: Number of results

        Returns:
            str: Hash key for caching
        """
        cache_input = f"{query.strip().lower()}:k={k}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def _calculate_query_hash_with_position(self, query: str, k: int, position_filter: Optional[List[float]]) -> str:
        """Generate cache key for query with position filter

        Args:
            query: Search query
            k: Number of results
            position_filter: Optional [min_pos, max_pos] range

        Returns:
            str: Hash key for caching
        """
        if position_filter:
            cache_input = f"{query.strip().lower()}:k={k}:pos=[{position_filter[0]:.4f},{position_filter[1]:.4f}]"
        else:
            cache_input = f"{query.strip().lower()}:k={k}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def _apply_temporal_filter(self, results: List[SearchResult], position_filter: List[float]) -> List[SearchResult]:
        """Filter search results by relative_position metadata

        Args:
            results: Raw search results
            position_filter: [min_position, max_position] range (0.0-1.0)

        Returns:
            List[SearchResult]: Filtered results within position range
        """
        if not position_filter or len(position_filter) != 2:
            return results

        min_pos, max_pos = position_filter
        filtered = []

        for result in results:
            # Extract relative_position from metadata
            relative_pos = result.metadata.get("relative_position")

            if relative_pos is None:
                # If no position metadata, skip this result
                logger.warning(f"[temporal_filter] Document missing relative_position metadata: {result.content[:50]}")
                continue

            # Check if within range
            if min_pos <= relative_pos <= max_pos:
                filtered.append(result)

        return filtered

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove semantic and text duplicates from search results

        Args:
            results: Raw search results

        Returns:
            List[SearchResult]: Deduplicated results
        """
        if not self.config.DEDUPLICATION_ENABLED or len(results) <= 1:
            return results

        deduplicated = []
        seen_hashes = set()

        for result in results:
            # Text-based deduplication
            content_hash = hashlib.md5(result.content.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue

            # Line range overlap detection
            overlap_found = False
            for existing in deduplicated:
                if self._has_range_overlap(result.line_range, existing.line_range):
                    # Keep the one with better score
                    if result.score < existing.score:  # Lower score is better
                        # Replace existing with current
                        deduplicated.remove(existing)
                        break
                    else:
                        overlap_found = True
                        break

            if not overlap_found:
                deduplicated.append(result)
                seen_hashes.add(content_hash)

        # Sort by score (lower is better)
        deduplicated.sort(key=lambda x: x.score)

        logger.debug(f"[FAISSRetriever] Deduplicated {len(results)} -> {len(deduplicated)} results")
        return deduplicated

    def _has_range_overlap(self, range1: List[int], range2: List[int],
                          overlap_threshold: float = 0.5) -> bool:
        """Check if two line ranges have significant overlap

        Args:
            range1: First range [start, end]
            range2: Second range [start, end]
            overlap_threshold: Minimum overlap ratio to consider significant

        Returns:
            bool: True if ranges overlap significantly
        """
        if not range1 or not range2 or len(range1) < 2 or len(range2) < 2:
            return False

        start1, end1 = range1[0], range1[1]
        start2, end2 = range2[0], range2[1]

        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start >= overlap_end:
            return False

        overlap_size = overlap_end - overlap_start
        range1_size = end1 - start1
        range2_size = end2 - start2

        if range1_size == 0 or range2_size == 0:
            return False

        overlap_ratio1 = overlap_size / range1_size
        overlap_ratio2 = overlap_size / range2_size

        return max(overlap_ratio1, overlap_ratio2) >= overlap_threshold

    def _search_single_subquery(self, subquery: Dict[str, Any]) -> SubqueryResult:
        """Process individual subquery with comprehensive error handling

        Args:
            subquery: Subquery dictionary from planner

        Returns:
            SubqueryResult: Structured search result
        """
        query_id = subquery.get("id", "unknown")
        query_text = subquery.get("text", "").strip()

        logger.debug(f"[FAISSRetriever] Processing subquery {query_id}: {query_text[:100]}")

        if not query_text:
            return SubqueryResult(
                query_id=query_id,
                query_text=query_text,
                results=[],
                search_time=0.0,
                result_count=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                error="Empty query text"
            )

        start_time = time.time()

        try:
            # Adaptive k calculation
            base_k = self.config.TOP_K_DEFAULT
            optimal_k = self._adaptive_k_calculation(query_text, base_k)

            # 🔍 Check for temporal constraint (position_filter)
            position_filter = subquery.get("position_filter")
            if position_filter:
                logger.info(f"[FAISSRetriever] Temporal filtering for {query_id}: position range [{position_filter[0]:.4f}, {position_filter[1]:.4f}]")

            # Check LRU+TTL cache (include position_filter in cache key)
            cache_key = self._calculate_query_hash_with_position(query_text, optimal_k, position_filter)
            cached_results = self.query_cache.get(cache_key)
            if cached_results is not None:
                self.total_cache_hits += 1
                search_time = time.time() - start_time

                logger.debug(f"[FAISSRetriever] Cache hit for {query_id}")

                return self._create_subquery_result(
                    query_id, query_text, cached_results, search_time, cache_hit=True
                )

            # Execute search without threshold filtering - let LLM evaluate quality later
            raw_results = self.search_engine.search(
                query=query_text,
                k=optimal_k,
                score_threshold=None  # Remove threshold filtering
            )

            # 🕒 Apply temporal filtering if position_filter is specified
            if position_filter:
                filtered_results = self._apply_temporal_filter(raw_results, position_filter)
                logger.debug(f"[FAISSRetriever] Temporal filter: {len(raw_results)} -> {len(filtered_results)} results")
                raw_results = filtered_results

            # Deduplicate results
            final_results = self._deduplicate_results(raw_results)

            # LRU+TTL Cache with automatic eviction
            self.query_cache.put(cache_key, final_results)

            search_time = time.time() - start_time
            self.search_times.append(search_time)
            self.total_searches += 1

            logger.debug(f"[FAISSRetriever] Found {len(final_results)} results for {query_id} (cached: {self.query_cache.size()}/{self.config.CACHE_MAX_SIZE})")

            return self._create_subquery_result(
                query_id, query_text, final_results, search_time, cache_hit=False
            )

        except Exception as e:
            search_time = time.time() - start_time
            error_msg = f"Search failed: {str(e)}"
            logger.error(f"[FAISSRetriever] Error in {query_id}: {error_msg}")

            return SubqueryResult(
                query_id=query_id,
                query_text=query_text,
                results=[],
                search_time=search_time,
                result_count=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                error=error_msg
            )

    def _create_subquery_result(self, query_id: str, query_text: str,
                               results: List[SearchResult], search_time: float,
                               cache_hit: bool = False) -> SubqueryResult:
        """Create structured subquery result with statistics

        Args:
            query_id: Query identifier
            query_text: Original query text
            results: Search results
            search_time: Time taken for search
            cache_hit: Whether result came from cache

        Returns:
            SubqueryResult: Complete result structure
        """
        if not results:
            return SubqueryResult(
                query_id=query_id,
                query_text=query_text,
                results=results,
                search_time=search_time,
                result_count=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                cache_hit=cache_hit
            )

        scores = [r.score for r in results]
        return SubqueryResult(
            query_id=query_id,
            query_text=query_text,
            results=results,
            search_time=search_time,
            result_count=len(results),
            avg_score=sum(scores) / len(scores),
            max_score=max(scores),
            min_score=min(scores),
            cache_hit=cache_hit
        )

    def search_subqueries(self, subqueries: List[Dict[str, Any]]) -> Tuple[Dict[str, SubqueryResult], SearchStats]:
        """Batch process multiple subqueries with performance tracking

        Args:
            subqueries: List of subquery dictionaries from planner

        Returns:
            Tuple[Dict[str, SubqueryResult], SearchStats]: Results and statistics
        """
        if not self._initialize_search_engine():
            logger.error("[FAISSRetriever] Search engine not available")
            return {}, self._create_empty_stats()

        logger.info(f"[FAISSRetriever] Processing {len(subqueries)} subqueries")
        start_time = time.time()

        results = {}

        # Process subqueries (sequential for now, can add parallel processing later)
        for subquery in subqueries:
            query_id = subquery.get("id", f"query_{len(results)}")
            result = self._search_single_subquery(subquery)
            results[query_id] = result

        total_time = time.time() - start_time
        stats = self._calculate_search_stats(results, total_time)

        logger.info(f"[FAISSRetriever] Completed batch search in {total_time:.2f}s")
        logger.info(f"[FAISSRetriever] Results: {stats.total_results} items, cache hits: {stats.cache_hits}")

        return results, stats

    def search_specific_subqueries(
        self,
        all_subqueries: List[Dict[str, Any]],
        target_query_ids: List[str]
    ) -> Tuple[Dict[str, SubqueryResult], SearchStats]:
        """Search only specific subqueries by ID (for partial reconstruction)

        Args:
            all_subqueries: Complete list of subqueries from planner
            target_query_ids: List of query IDs to search for

        Returns:
            Tuple[Dict[str, SubqueryResult], SearchStats]: Results for target queries only
        """
        if not self._initialize_search_engine():
            logger.error("[FAISSRetriever] Search engine not available")
            return {}, self._create_empty_stats()

        # Filter subqueries to only target IDs
        target_subqueries = [
            sq for sq in all_subqueries
            if sq.get("id") in target_query_ids
        ]

        if not target_subqueries:
            logger.warning(f"[FAISSRetriever] No subqueries found for target IDs: {target_query_ids}")
            return {}, self._create_empty_stats()

        logger.info(f"[FAISSRetriever] Partial search: Processing {len(target_subqueries)} specific subqueries")
        start_time = time.time()

        results = {}

        # Process only the target subqueries
        for subquery in target_subqueries:
            query_id = subquery.get("id")
            result = self._search_single_subquery(subquery)
            results[query_id] = result

        total_time = time.time() - start_time
        stats = self._calculate_search_stats(results, total_time)

        logger.info(f"[FAISSRetriever] Completed partial search in {total_time:.2f}s")
        logger.info(f"[FAISSRetriever] Partial results: {stats.total_results} items for {len(target_subqueries)} queries")

        return results, stats

    def _calculate_search_stats(self, results: Dict[str, SubqueryResult],
                               total_time: float) -> SearchStats:
        """Calculate comprehensive search statistics

        Args:
            results: Search results by query ID
            total_time: Total processing time

        Returns:
            SearchStats: Aggregated statistics
        """
        if not results:
            return self._create_empty_stats()

        total_results = sum(r.result_count for r in results.values())
        cache_hits = sum(1 for r in results.values() if r.cache_hit)
        cache_hit_rate = cache_hits / len(results) if results else 0.0

        # Calculate average score across all results
        all_scores = []
        for result in results.values():
            all_scores.extend([r.score for r in result.results])

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Query type distribution
        query_distribution = defaultdict(int)
        for result in results.values():
            words = len(result.query_text.split())
            if words <= self.config.SHORT_QUERY_WORDS:
                query_distribution["short"] += 1
            elif words >= self.config.LONG_QUERY_WORDS:
                query_distribution["long"] += 1
            else:
                query_distribution["medium"] += 1

        # Score distribution
        score_distribution = defaultdict(int)
        for score in all_scores:
            if score <= 0.2:
                score_distribution["excellent"] += 1
            elif score <= 0.5:
                score_distribution["good"] += 1
            elif score <= 0.8:
                score_distribution["fair"] += 1
            else:
                score_distribution["poor"] += 1

        # Performance metrics
        search_times = [r.search_time for r in results.values()]
        performance_metrics = {
            "avg_search_time": sum(search_times) / len(search_times) if search_times else 0.0,
            "max_search_time": max(search_times) if search_times else 0.0,
            "min_search_time": min(search_times) if search_times else 0.0,
            "results_per_second": total_results / total_time if total_time > 0 else 0.0
        }

        return SearchStats(
            total_queries=len(results),
            total_results=total_results,
            avg_score=avg_score,
            search_time=total_time,
            cache_hits=cache_hits,
            cache_hit_rate=cache_hit_rate,
            query_distribution=dict(query_distribution),
            score_distribution=dict(score_distribution),
            performance_metrics=performance_metrics
        )

    def _create_empty_stats(self) -> SearchStats:
        """Create empty statistics object for error cases"""
        return SearchStats(
            total_queries=0,
            total_results=0,
            avg_score=0.0,
            search_time=0.0,
            cache_hits=0,
            cache_hit_rate=0.0,
            query_distribution={},
            score_distribution={},
            performance_metrics={}
        )

# ─────────────────────────────────────────────────────────
# LangGraph Node Function
# ─────────────────────────────────────────────────────────

def merge_retrieval_results(
    existing_results: Dict[str, Any],
    new_results: Dict[str, Any],
    search_stats: SearchStats
) -> Dict[str, Any]:
    """Merge new search results with existing preserved results

    Args:
        existing_results: Previously stored results from successful subqueries
        new_results: New results from failed subquery reconstruction
        search_stats: Statistics from new search

    Returns:
        Dict[str, Any]: Merged results preserving successful queries
    """
    logger.info(f"[merge_results] Merging {len(new_results)} new results with {len(existing_results)} existing")

    # Start with existing results
    merged_results = existing_results.copy()

    # Add new results (will overwrite any existing results for same query IDs)
    merged_results.update(new_results)

    logger.info(f"[merge_results] Final merged results: {len(merged_results)} total")
    return merged_results


def merge_search_stats(
    existing_stats: Dict[str, Any],
    new_stats: SearchStats,
    merge_type: str = "incremental"
) -> Dict[str, Any]:
    """Merge search statistics from partial searches

    Args:
        existing_stats: Previously stored search statistics
        new_stats: New statistics from partial search
        merge_type: Type of merge ("incremental" or "replace")

    Returns:
        Dict[str, Any]: Merged statistics
    """
    if merge_type == "replace" or not existing_stats:
        return asdict(new_stats)

    # Incremental merge
    merged_stats = existing_stats.copy()

    # Update counts and times incrementally
    merged_stats["total_queries"] = existing_stats.get("total_queries", 0) + new_stats.total_queries
    merged_stats["total_results"] = existing_stats.get("total_results", 0) + new_stats.total_results
    merged_stats["search_time"] = existing_stats.get("search_time", 0.0) + new_stats.search_time
    merged_stats["cache_hits"] = existing_stats.get("cache_hits", 0) + new_stats.cache_hits

    # Recalculate derived metrics
    if merged_stats["total_queries"] > 0:
        merged_stats["cache_hit_rate"] = merged_stats["cache_hits"] / merged_stats["total_queries"]
    else:
        merged_stats["cache_hit_rate"] = 0.0

    # Average score needs weighted calculation
    existing_total = existing_stats.get("total_results", 0)
    new_total = new_stats.total_results
    total_combined = existing_total + new_total

    if total_combined > 0:
        existing_avg = existing_stats.get("avg_score", 0.0)
        merged_stats["avg_score"] = (
            (existing_avg * existing_total + new_stats.avg_score * new_total) / total_combined
        )
    else:
        merged_stats["avg_score"] = 0.0

    # Merge distributions
    for dist_key in ["query_distribution", "score_distribution"]:
        existing_dist = existing_stats.get(dist_key, {})
        new_dist = getattr(new_stats, dist_key, {})

        merged_dist = existing_dist.copy()
        for key, value in new_dist.items():
            merged_dist[key] = merged_dist.get(key, 0) + value

        merged_stats[dist_key] = merged_dist

    return merged_stats


def retrieve_node(state: AgentState) -> AgentState:
    """LangGraph-compatible retrieval node for processing planner subqueries

    This function supports both full and partial search modes:
    - Full search: Process all subqueries from planner (initial retrieval)
    - Partial search: Process only reconstructed subqueries and merge with existing results

    Args:
        state: Current agent state from LangGraph

    Returns:
        AgentState: Updated state with retrieval results
    """
    logger.info("[retrieve_node] Starting FAISS retrieval processing")

    # Extract existing data
    messages: List[BaseMessage] = state.get("messages", [])

    # MetadataManager를 사용한 안전한 메타데이터 조회
    planner_data = MetadataManager.safe_get_metadata(state, "planner")
    if not planner_data:
        logger.warning("[retrieve_node] No planner data found in metadata")
        return state

    plan_json = planner_data.get("last_plan_json", {})
    subqueries = plan_json.get("subqueries", [])

    if not subqueries:
        logger.warning("[retrieve_node] No subqueries found in planner data")
        return state

    # Check for partial search mode (reconstructed subqueries from replanner)
    replanner_data = MetadataManager.safe_get_metadata(state, "replanner") or {}
    is_partial_search = replanner_data.get("status") == "completed"

    if is_partial_search:
        # Partial search mode: only process reconstructed subqueries
        reconstructions = replanner_data.get("reconstructions", [])
        reconstructed_query_ids = [recon["subquery_id"] for recon in reconstructions]

        if not reconstructed_query_ids:
            logger.warning("[retrieve_node] Partial search mode but no reconstructed queries found")
            return state

        logger.info(f"[retrieve_node] Partial search mode: Processing {len(reconstructed_query_ids)} reconstructed subqueries")
        search_mode = "partial"
    else:
        # Full search mode: process all subqueries
        logger.info(f"[retrieve_node] Full search mode: Processing {len(subqueries)} subqueries")
        search_mode = "full"

    # Initialize retriever with max_docs from experiment config
    try:
        # ✅ MetadataManager를 사용한 실험 설정 조회
        metadata: Dict[str, Any] = state.get("metadata", {}) or {}
        experiment_config = metadata.get("experiment_config", {})
        max_docs = experiment_config.get("max_docs", None)

        # Create retrieval config with max_docs override
        retrieval_config = RetrievalConfig(max_docs=max_docs)
        retriever = FAISSRetriever(config=retrieval_config)

        if max_docs is not None:
            logger.info(f"[retrieve_node] Using max_docs={max_docs} from experiment config")

        if search_mode == "partial":
            # Execute partial retrieval for reconstructed queries only
            results_by_qid, search_stats = retriever.search_specific_subqueries(
                all_subqueries=subqueries,
                target_query_ids=reconstructed_query_ids
            )
        else:
            # Execute full batch retrieval
            results_by_qid, search_stats = retriever.search_subqueries(subqueries)

        # Convert results to serializable format
        new_serializable_results = {}
        for qid, result in results_by_qid.items():
            new_serializable_results[qid] = {
                "query_id": result.query_id,
                "query_text": result.query_text,
                "results": [asdict(r) for r in result.results],
                "search_time": result.search_time,
                "result_count": result.result_count,
                "avg_score": result.avg_score,
                "max_score": result.max_score,
                "min_score": result.min_score,
                "cache_hit": result.cache_hit,
                "error": result.error
            }

        # Handle result merging for partial search
        if search_mode == "partial":
            # MetadataManager를 사용하여 기존 결과 조회
            existing_retriever_data = MetadataManager.safe_get_metadata(state, "faiss_retriever") or {}
            existing_results = existing_retriever_data.get("results_by_qid", {})
            existing_stats = existing_retriever_data.get("search_stats", {})

            # Merge results
            final_serializable_results = merge_retrieval_results(
                existing_results, new_serializable_results, search_stats
            )

            # Merge statistics
            final_search_stats = merge_search_stats(existing_stats, search_stats, "incremental")

            summary_content = (
                f"FAISS Partial Retrieval Complete\n"
                f"Reconstructed: {len(reconstructed_query_ids)} subqueries\n"
                f"New Results: {search_stats.total_results} items\n"
                f"Total Results: {final_search_stats.get('total_results', 0)} items\n"
                f"Average Score: {search_stats.avg_score:.4f}\n"
                f"Search Time: {search_stats.search_time:.2f}s"
            )
        else:
            # Full search: use results as-is
            final_serializable_results = new_serializable_results
            final_search_stats = asdict(search_stats)

            summary_content = (
                f"FAISS Full Retrieval Complete\n"
                f"Processed: {len(subqueries)} subqueries\n"
                f"Found: {search_stats.total_results} total results\n"
                f"Average Score: {search_stats.avg_score:.4f}\n"
                f"Cache Hits: {search_stats.cache_hits}/{len(subqueries)}\n"
                f"Search Time: {search_stats.search_time:.2f}s"
            )

        # Create queries mapping from final results
        queries_by_qid = {}
        for qid, result_data in final_serializable_results.items():
            queries_by_qid[qid] = result_data.get("query_text", "")

        # Store in metadata.faiss_retriever namespace
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
            "success": True
        }

        # Update metadata safely
        new_metadata = metadata.copy()
        new_metadata["faiss_retriever"] = retriever_metadata
        new_metadata["last_agent"] = "faiss_retriever"

        # 🔧 재구성된 쿼리 사용 완료 후 replanner 상태 초기화 (루프 방지)
        if search_mode == "partial" and replanner_data.get("status") == "completed":
            logger.info("[retrieve_node] Marking reconstructed queries as consumed to prevent routing loops")
            new_metadata["replanner"] = replanner_data.copy()
            new_metadata["replanner"]["status"] = "queries_consumed"
            new_metadata["replanner"]["consumed_timestamp"] = datetime.now(timezone.utc).isoformat()
            new_metadata["replanner"]["consumption_note"] = "Reconstructed queries have been used for retrieval"

        retrieval_message = AIMessage(
            content=summary_content,
            additional_kwargs={
                "agent_name": "faiss_retriever",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "search_mode": search_mode,
                "retrieval_stats": final_search_stats if search_mode == "full" else asdict(search_stats)
            }
        )

        new_messages = messages + [retrieval_message]

        logger.info(f"[retrieve_node] Successfully completed {search_mode} search")
        if search_mode == "partial":
            logger.info(f"[retrieve_node] Partial results: {search_stats.total_results} new, {final_search_stats.get('total_results', 0)} total")
        else:
            logger.info(f"[retrieve_node] Full results: {search_stats.total_results}, Avg score: {search_stats.avg_score:.4f}")

        # Return updated state
        return type(state)(
            messages=new_messages,
            context_messages=state.get("context_messages", []),
            metadata=new_metadata,
            current_agent="faiss_retriever",
            session_id=state.get("session_id", "")
        )

    except Exception as e:
        error_msg = f"FAISS retrieval failed: {str(e)}"
        logger.error(f"[retrieve_node] {error_msg}")

        # Store error in metadata
        new_metadata = metadata.copy()
        new_metadata["faiss_retriever"] = {
            "success": False,
            "error": error_msg,
            "search_mode": search_mode if 'search_mode' in locals() else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        new_metadata["last_agent"] = "faiss_retriever"

        # Create error message
        error_message = AIMessage(
            content=f"Retrieval Error: {error_msg}",
            additional_kwargs={
                "agent_name": "faiss_retriever",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": True
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
# Utility Functions
# ─────────────────────────────────────────────────────────

def format_retrieval_results(retriever_metadata: Dict[str, Any]) -> str:
    """Format retrieval results for human-readable display

    Args:
        retriever_metadata: Metadata from faiss_retriever namespace

    Returns:
        str: Formatted results string
    """
    if not retriever_metadata.get("success", False):
        return f"Retrieval failed: {retriever_metadata.get('error', 'Unknown error')}"

    results_by_qid = retriever_metadata.get("results_by_qid", {})
    search_stats = retriever_metadata.get("search_stats", {})

    lines = ["=== FAISS Retrieval Results ===\n"]

    for qid, result_data in results_by_qid.items():
        query_text = result_data.get("query_text", "")
        result_count = result_data.get("result_count", 0)
        avg_score = result_data.get("avg_score", 0.0)
        cache_hit = result_data.get("cache_hit", False)

        lines.append(f"Query {qid}: {query_text}")
        lines.append(f"  Results: {result_count} items")
        lines.append(f"  Avg Score: {avg_score:.4f}")
        lines.append(f"  Cache Hit: {'Yes' if cache_hit else 'No'}")

        # Show top 3 results
        results = result_data.get("results", [])
        for i, result in enumerate(results[:3], 1):
            content_preview = result.get("content", "")[:100]
            score = result.get("score", 0.0)
            lines.append(f"    {i}. Score: {score:.4f} | {content_preview}...")

        lines.append("")

    # Add summary statistics
    lines.append("=== Search Statistics ===")
    lines.append(f"Total Queries: {search_stats.get('total_queries', 0)}")
    lines.append(f"Total Results: {search_stats.get('total_results', 0)}")
    lines.append(f"Average Score: {search_stats.get('avg_score', 0.0):.4f}")
    lines.append(f"Search Time: {search_stats.get('search_time', 0.0):.2f}s")
    lines.append(f"Cache Hit Rate: {search_stats.get('cache_hit_rate', 0.0):.1%}")

    return "\n".join(lines)

def get_retrieval_results_for_query(state: AgentState, query_id: str) -> Optional[List[SearchResult]]:
    """Extract search results for specific query ID from state

    MetadataManager를 사용하여 안전하게 검색 결과를 조회합니다.

    Args:
        state: Agent state containing retrieval results
        query_id: Query identifier to look up

    Returns:
        Optional[List[SearchResult]]: Search results or None if not found
    """
    # MetadataManager를 사용한 안전한 메타데이터 조회
    retriever_data = MetadataManager.safe_get_metadata(state, "faiss_retriever")

    if not retriever_data or not retriever_data.get("success", False):
        return None

    results_by_qid = retriever_data.get("results_by_qid", {})
    result_data = results_by_qid.get(query_id)

    if not result_data:
        return None

    # Convert back to SearchResult objects
    raw_results = result_data.get("results", [])
    search_results = [
        SearchResult(
            rank=raw_result.get("rank", 0),
            score=raw_result.get("score", 0.0),
            content=raw_result.get("content", ""),
            metadata=raw_result.get("metadata", {}),
            line_range=raw_result.get("line_range", []),
            time_start=raw_result.get("time_start", ""),
            time_end=raw_result.get("time_end", "")
        )
        for raw_result in raw_results
    ]

    return search_results

# ─────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────

__all__ = [
    "FAISSRetriever",
    "SubqueryResult",
    "SearchStats",
    "RetrievalConfig",
    "retrieve_node",
    "format_retrieval_results",
    "get_retrieval_results_for_query"
]