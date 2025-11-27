
"""
Memory-Optimized FAISS Retriever - Drop-in replacement for production use
========================================================================

Replaces the original FAISSRetriever with memory-safe implementation that prevents
CUDA out of memory errors through singleton pattern and aggressive cleanup.

Key Features:
- Singleton FAISS engine reuse across all retrieve_node() calls
- Real-time memory pressure monitoring and cleanup
- Automatic fallback and recovery mechanisms
- Production-ready memory management
- Drop-in compatibility with existing retrieve_node()

Architecture:
- MemoryOptimizedRetriever: Memory-safe replacement for FAISSRetriever
- Uses MemoryOptimizedFAISSManager singleton for engine management
- Maintains identical API for seamless replacement
"""

from __future__ import annotations
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from ..memory.memory_manager import MemoryOptimizedFAISSManager
from .retrieve import (
    SubqueryResult, SearchStats, RetrievalConfig,
    SearchResult, logger
)

# ─────────────────────────────────────────────────────────
# Memory-Optimized FAISS Retriever
# ─────────────────────────────────────────────────────────

class MemoryOptimizedRetriever:
    """
    Memory-safe FAISS retriever that prevents CUDA OOM through singleton engine reuse

    Drop-in replacement for FAISSRetriever with identical API but superior memory management.
    Uses singleton pattern to reuse FAISS engine across all retrieval operations.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize memory-optimized retriever

        Args:
            config: Optional configuration object, uses defaults if None
        """
        self.config = config or RetrievalConfig()
        self.manager = MemoryOptimizedFAISSManager()
        self.query_cache: Dict[str, List[SearchResult]] = {}
        self.stats_cache: Dict[str, Any] = {}
        self.initialized = False

        # Performance tracking (same as original)
        self.total_searches = 0
        self.total_cache_hits = 0
        self.search_times: List[float] = []

        logger.info(f"[MemoryOptimizedRetriever] Initializing with index: {self.config.INDEX_PATH}")

    def _initialize_search_engine(self) -> bool:
        """
        Initialize the FAISS search engine with memory-safe singleton

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.initialized:
            return True

        try:
            logger.info(f"[MemoryOptimizedRetriever] Initializing singleton engine")

            # Log memory status before initialization
            self.manager.memory_monitor.log_memory_status("Before Retriever Init")

            # Configure engine parameters
            engine_config = {
                "index_path": self.config.INDEX_PATH,
                "model_name": self.config.MODEL_NAME,
                "device": self.config.DEVICE,
                "use_fp16": True,
                "max_length": self.config.QUERY_MAX_LENGTH
            }

            # Initialize engine through singleton manager
            success = self.manager.initialize_engine(engine_config)

            if success:
                self.initialized = True
                logger.info("[MemoryOptimizedRetriever] Singleton engine initialized successfully")

                # Log memory status after initialization
                self.manager.memory_monitor.log_memory_status("After Retriever Init")

                return True
            else:
                logger.error("[MemoryOptimizedRetriever] Failed to initialize singleton engine")
                return False

        except Exception as e:
            logger.error(f"[MemoryOptimizedRetriever] Engine initialization error: {e}")
            self.initialized = False
            return False

    def _adaptive_k_calculation(self, query: str, base_k: int) -> int:
        """Calculate adaptive k value based on query characteristics (same as original)"""
        if not self.config.ADAPTIVE_K_ENABLED:
            return base_k

        words = query.split()
        word_count = len(words)

        # Fixed k value - no adaptive adjustment based on query length or keywords
        k_multiplier = 1.0

        adjusted_k = int(base_k * k_multiplier)
        return min(max(adjusted_k, 1), self.config.TOP_K_MAX)

    def _calculate_query_hash(self, query: str, k: int) -> str:
        """Generate cache key for query (same as original)"""
        import hashlib
        cache_input = f"{query.strip().lower()}:k={k}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove semantic and text duplicates from search results (same as original)"""
        if not self.config.DEDUPLICATION_ENABLED or len(results) <= 1:
            return results

        import hashlib
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

        logger.debug(f"[MemoryOptimizedRetriever] Deduplicated {len(results)} -> {len(deduplicated)} results")
        return deduplicated

    def _has_range_overlap(self, range1: List[int], range2: List[int],
                          overlap_threshold: float = 0.5) -> bool:
        """Check if two line ranges have significant overlap (same as original)"""
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

    def _build_position_filter(self, position_filter: Optional[List[float]],
                               padding_ratio: float = 0.015) -> Optional[Dict[str, Any]]:
        """
        Convert position_filter [start, end] to FAISS metadata filter with padding

        ⚠️ IMPORTANT: FAISS indexes use 100-line chunks, so narrow filters may miss chunks.
        This function adds PADDING to ensure chunks overlapping with the target range are included.

        Example Problem:
        - Ground truth range: Lines 439-446 (relative: 0.0506-0.0514)
        - FAISS chunk containing it: Lines 401-500 (relative: 0.0462-0.0576)
        - Without padding: 0.0506-0.0514 filter misses 0.0462 chunk start!
        - With padding: 0.0391-0.0629 filter includes the chunk ✓

        Args:
            position_filter: [start_position, end_position] (e.g., [0.6121, 0.6621])
            padding_ratio: Padding added to both sides (default 0.015 = 1.5% of log)
                          For 8684 lines, 0.015 = ~130 lines = 1.3 chunks of padding

        Returns:
            FAISS-compatible metadata filter dict with padding, or None if no filter needed

        Example:
            Input: [0.0506, 0.0514], padding_ratio=0.015
            Output: {
                "relative_position": {
                    "$gte": 0.0356,  # 0.0506 - 0.015
                    "$lte": 0.0664   # 0.0514 + 0.015
                }
            }
        """
        if not position_filter or len(position_filter) != 2:
            return None

        start, end = position_filter

        # Validate range
        if start < 0 or end > 1.0 or start >= end:
            logger.warning(f"[_build_position_filter] Invalid position_filter: {position_filter}, ignoring")
            return None

        # ✨ Add padding to handle FAISS chunking (100-line chunks)
        # Padding ensures we don't miss chunks that overlap with target range
        padded_start = max(0.0, start - padding_ratio)
        padded_end = min(1.0, end + padding_ratio)

        # FAISS metadata filter format for LangChain
        filter_dict = {
            "relative_position": {
                "$gte": float(padded_start),  # greater than or equal
                "$lte": float(padded_end)     # less than or equal
            }
        }

        logger.info(f"[_build_position_filter] Original range: [{start:.4f}, {end:.4f}]")
        logger.info(f"[_build_position_filter] Padded range (+{padding_ratio:.4f}): [{padded_start:.4f}, {padded_end:.4f}]")
        return filter_dict

    def _apply_diversity_filter(
        self,
        results: Dict[str, SubqueryResult],
        subqueries: List[Dict[str, Any]]
    ) -> Dict[str, SubqueryResult]:
        """
        Ensure each subquery selects non-overlapping documents.

        Process subqueries sequentially (Q1 → Q2 → Q3...) and for each query,
        select top-k documents that don't overlap with previously selected documents.

        Args:
            results: Query results from parallel execution
            subqueries: Original subquery list with ordering

        Returns:
            Updated results with diversity-filtered documents
        """
        # Track selected line ranges across all queries
        selected_ranges = []

        # Top-k to select per query (same as optimal_k)
        top_k = self.config.TOP_K_DEFAULT

        # Sort subqueries by ID to ensure consistent processing order (Q1, Q2, Q3...)
        sorted_queries = sorted(subqueries, key=lambda x: x.get("id", ""))

        logger.info(f"[_apply_diversity_filter] Applying diversity filter across {len(sorted_queries)} subqueries (top_k={top_k})")

        updated_results = {}
        for subquery in sorted_queries:
            query_id = subquery.get("id", "")
            if query_id not in results:
                continue

            original_result = results[query_id]
            original_docs = original_result.results

            if not original_docs:
                updated_results[query_id] = original_result
                continue

            # Select non-overlapping documents
            selected_docs = []
            for doc in original_docs:
                # Check if this document overlaps with any previously selected document
                has_overlap = False
                for selected_range in selected_ranges:
                    if self._has_range_overlap(doc.line_range, selected_range):
                        has_overlap = True
                        break

                if not has_overlap:
                    selected_docs.append(doc)
                    selected_ranges.append(doc.line_range)

                    # Stop when we have enough documents
                    if len(selected_docs) >= top_k:
                        break

            # Log diversity filtering effect
            if len(selected_docs) < len(original_docs):
                logger.info(
                    f"[_apply_diversity_filter] {query_id}: {len(original_docs)} candidates → selected {len(selected_docs)} unique docs "
                    f"(skipped {len(original_docs) - len(selected_docs)} overlapping with previous queries)"
                )
            else:
                logger.info(f"[_apply_diversity_filter] {query_id}: selected {len(selected_docs)} docs (no overlap with previous queries)")

            # Create updated result with selected documents
            updated_results[query_id] = SubqueryResult(
                query_id=original_result.query_id,
                query_text=original_result.query_text,
                results=selected_docs,
                search_time=original_result.search_time,
                result_count=len(selected_docs),
                avg_score=sum(d.score for d in selected_docs) / len(selected_docs) if selected_docs else 0.0,
                max_score=max(d.score for d in selected_docs) if selected_docs else 0.0,
                min_score=min(d.score for d in selected_docs) if selected_docs else 0.0,
                cache_hit=original_result.cache_hit
            )

        return updated_results

    def _search_single_subquery(self, subquery: Dict[str, Any]) -> SubqueryResult:
        """
        Process individual subquery with memory-safe singleton engine

        Args:
            subquery: Subquery dictionary from planner

        Returns:
            SubqueryResult: Structured search result
        """
        query_id = subquery.get("id", "unknown")
        query_text = subquery.get("text", "").strip()

        # ✨ Extract temporal_constraint from subquery (new field from improved prompt)
        temporal_constraint = subquery.get("temporal_constraint", "")
        phase = subquery.get("phase", "")

        logger.debug(f"[MemoryOptimizedRetriever] Processing subquery {query_id}: {query_text[:100]}")
        if temporal_constraint:
            logger.debug(f"[MemoryOptimizedRetriever] {query_id} temporal_constraint: {temporal_constraint}, phase: {phase}")

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

            # Check cache
            cache_key = self._calculate_query_hash(query_text, optimal_k)
            if cache_key in self.query_cache:
                results = self.query_cache[cache_key]
                self.total_cache_hits += 1
                search_time = time.time() - start_time

                logger.debug(f"[MemoryOptimizedRetriever] Cache hit for {query_id}")

                return self._create_subquery_result(
                    query_id, query_text, results, search_time, cache_hit=True
                )

            # ✨ NEW: Extract position_filter from subquery for PRE-FILTERING
            position_filter = subquery.get("position_filter")
            metadata_filter = self._build_position_filter(position_filter)

            # Execute search through memory-safe singleton manager
            # ✨ PRE-FILTERING: FAISS searches only within position_filter range
            # This replaces the old POST-FILTERING approach (search all → filter later)
            if metadata_filter:
                search_k = self.config.TOP_K_MAX  # Higher k for better coverage within range
                final_k = optimal_k
                logger.info(f"[MemoryOptimizedRetriever] {query_id} PRE-FILTERING mode: search_k={search_k}, position_filter={position_filter}")
                logger.info(f"[MemoryOptimizedRetriever] {query_id} metadata_filter passed to FAISS: {metadata_filter}")
            elif temporal_constraint:
                # Fallback: Use old temporal filtering if no position_filter
                search_k = self.config.TOP_K_MAX
                final_k = optimal_k
                logger.info(f"[MemoryOptimizedRetriever] {query_id} temporal mode (fallback): search_k={search_k}, constraint={temporal_constraint}")
            else:
                search_k = optimal_k
                final_k = optimal_k

            # ✨ Pass metadata_filter to FAISS for PRE-FILTERING
            raw_results = self.manager.search(
                query=query_text,
                k=search_k,
                score_threshold=1.0,  # Filter weak semantic matches (1.0 = most lenient, include all results)
                filter=metadata_filter  # ← NEW: Pre-filter by position BEFORE search
            )
            logger.info(f"[MemoryOptimizedRetriever] {query_id} FAISS returned {len(raw_results)} results after filtering")

            # ✨ Apply temporal filtering if constraint exists
            if temporal_constraint and raw_results:
                logger.info(f"[MemoryOptimizedRetriever] {query_id} applying temporal filter: {temporal_constraint}")
                logger.info(f"[MemoryOptimizedRetriever] {query_id} raw_results before filter: {len(raw_results)} documents")

                # 필터링 전 모든 문서의 relative_position과 score 분포 로깅
                positions = []
                scores = []
                for i, result in enumerate(raw_results):
                    rel_pos = result.metadata.get("relative_position")
                    line_num = result.metadata.get("line_number")
                    range_info = result.metadata.get("Range")
                    score = result.score

                    positions.append(f"{rel_pos:.4f}" if rel_pos is not None else "None")
                    scores.append(f"{score:.4f}")

                    if i < 5:  # 처음 5개는 상세 로깅 (score 포함)
                        rel_pos_str = f"{rel_pos:.4f}" if rel_pos is not None else "None"
                        logger.info(f"[MemoryOptimizedRetriever] {query_id} doc[{i}]: "
                                   f"score={score:.4f}, rel_pos={rel_pos_str}, line={line_num}, Range={range_info}")

                logger.info(f"[MemoryOptimizedRetriever] {query_id} All scores: [{', '.join(scores)}]")
                logger.info(f"[MemoryOptimizedRetriever] {query_id} All relative_positions: [{', '.join(positions)}]")

                # EARLY 영역(0-30%) 문서들의 score 분석
                early_docs = [(i, r) for i, r in enumerate(raw_results)
                             if r.metadata.get("relative_position", 1.0) <= 0.3]
                if early_docs and temporal_constraint and "EARLY" in temporal_constraint.upper():
                    logger.info(f"[MemoryOptimizedRetriever] {query_id} EARLY region (0-30%) documents found: {len(early_docs)}")
                    for i, (idx, r) in enumerate(early_docs[:5]):
                        logger.info(f"[MemoryOptimizedRetriever] {query_id} EARLY doc rank={idx}, score={r.score:.4f}, "
                                   f"rel_pos={r.metadata.get('relative_position'):.4f}, Range={r.metadata.get('Range')}")

                filtered_results = self._apply_temporal_filter(raw_results, temporal_constraint)
                logger.info(f"[MemoryOptimizedRetriever] {query_id} temporal filter: {len(raw_results)} → {len(filtered_results)} results")

                if len(filtered_results) == 0:
                    logger.warning(f"[MemoryOptimizedRetriever] {query_id} ⚠️ Temporal filter removed ALL results! "
                                 f"constraint={temporal_constraint}, "
                                 f"original_count={len(raw_results)}")

                # Keep all filtered results for diversity filtering later
                # Diversity filter will select top-k while avoiding overlap
                raw_results = filtered_results

            # Deduplicate results
            final_results = self._deduplicate_results(raw_results)

            # Cache results
            if len(self.query_cache) < self.config.CACHE_MAX_SIZE:
                self.query_cache[cache_key] = final_results

            search_time = time.time() - start_time
            self.search_times.append(search_time)
            self.total_searches += 1

            logger.debug(f"[MemoryOptimizedRetriever] Found {len(final_results)} results for {query_id}")

            return self._create_subquery_result(
                query_id, query_text, final_results, search_time, cache_hit=False
            )

        except Exception as e:
            search_time = time.time() - start_time
            error_msg = f"Search failed: {str(e)}"
            logger.error(f"[MemoryOptimizedRetriever] Error in {query_id}: {error_msg}")

            # Log memory stats on error for debugging
            memory_stats = self.manager.get_memory_stats()
            logger.error(f"[MemoryOptimizedRetriever] Memory stats on error: {memory_stats}")

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
        """Create structured subquery result with statistics (same as original)"""
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

    async def search_subqueries(self, subqueries: List[Dict[str, Any]], provider: str = None) -> Tuple[Dict[str, SubqueryResult], SearchStats]:
        """
        Batch process multiple subqueries with ADAPTIVE EXECUTION (parallel or sequential)

        **ADAPTIVE PROCESSING**:
        - API models (Gemini/GPT/Llama): Parallel processing for speed
        - Local models (EXAONE): Sequential processing for memory safety

        Args:
            subqueries: List of subquery dictionaries from planner
            provider: LLM provider name (optional, for adaptive processing)

        Returns:
            Tuple[Dict[str, SubqueryResult], SearchStats]: Results and statistics
        """
        import asyncio

        if not self._initialize_search_engine():
            logger.error("[MemoryOptimizedRetriever] Search engine not available")
            return {}, self._create_empty_stats()

        # 🔧 Provider 기반 처리 전략 결정
        is_sequential = False
        if provider and provider.lower() in ["exaone"]:
            is_sequential = True
            logger.info(f"[MemoryOptimizedRetriever] Sequential mode for local model ({provider})")
        else:
            logger.info(f"[MemoryOptimizedRetriever] Parallel mode for API model ({provider or 'default'})")

        logger.info(f"[MemoryOptimizedRetriever] Processing {len(subqueries)} subqueries ({'SEQUENTIAL' if is_sequential else 'PARALLEL'})")

        # Log memory status before batch processing
        self.manager.memory_monitor.log_memory_status("Before Batch Processing")

        start_time = time.time()

        # ═══════════════════════════════════════════════════════
        # ADAPTIVE PROCESSING: Sequential or Parallel
        # ═══════════════════════════════════════════════════════
        if is_sequential:
            # Sequential processing for EXAONE (memory-safe)
            results = {}
            for subquery in subqueries:
                query_id = subquery.get("id", f"query_{hash(str(subquery))}")
                result = self._search_single_subquery(subquery)
                results[query_id] = result
            logger.info(f"[MemoryOptimizedRetriever] Sequential processing completed")
        else:
            # Parallel processing for API models (faster)
            async def process_single_async(subquery: Dict[str, Any]) -> Tuple[str, SubqueryResult]:
                """Async wrapper for single subquery processing"""
                query_id = subquery.get("id", f"query_{hash(str(subquery))}")
                # Run blocking I/O in thread pool
                result = await asyncio.to_thread(self._search_single_subquery, subquery)
                return query_id, result

            # Create parallel tasks for all subqueries
            tasks = [process_single_async(sq) for sq in subqueries]

            # Execute all tasks in parallel
            logger.info(f"[MemoryOptimizedRetriever] Executing {len(tasks)} parallel search tasks...")
            query_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            results = {}
            for item in query_results:
                if isinstance(item, Exception):
                    logger.error(f"[MemoryOptimizedRetriever] Parallel task failed: {item}")
                    continue

                query_id, result = item
                results[query_id] = result

        # ✨ Apply diversity filter: ensure each subquery selects non-overlapping documents
        results = self._apply_diversity_filter(results, subqueries)

        total_time = time.time() - start_time
        stats = self._calculate_search_stats(results, total_time)

        # Log final memory status and performance metrics
        self.manager.memory_monitor.log_memory_status("After Batch Processing")

        # Log manager statistics
        memory_stats = self.manager.get_memory_stats()
        mode_str = "Sequential" if is_sequential else "Parallel"
        logger.info(
            f"[MemoryOptimizedRetriever] ✓ {mode_str} batch completed in {total_time:.2f}s: "
            f"{stats.total_results} items, cache hits: {stats.cache_hits}, "
            f"GPU: {memory_stats.get('gpu_utilization_pct', 0):.1f}%"
        )

        return results, stats

    async def search_specific_subqueries(
        self,
        all_subqueries: List[Dict[str, Any]],
        target_query_ids: List[str],
        provider: str = None
    ) -> Tuple[Dict[str, SubqueryResult], SearchStats]:
        """
        Search only specific subqueries by ID with PARALLEL memory-safe processing

        Args:
            all_subqueries: Complete list of subqueries from planner
            target_query_ids: List of query IDs to search for

        Returns:
            Tuple[Dict[str, SubqueryResult], SearchStats]: Results for target queries only
        """
        if not self._initialize_search_engine():
            logger.error("[MemoryOptimizedRetriever] Search engine not available")
            return {}, self._create_empty_stats()

        # Filter subqueries to only target IDs
        target_subqueries = [
            sq for sq in all_subqueries
            if sq.get("id") in target_query_ids
        ]

        if not target_subqueries:
            logger.warning(f"[MemoryOptimizedRetriever] No subqueries found for target IDs: {target_query_ids}")
            return {}, self._create_empty_stats()

        logger.info(f"[MemoryOptimizedRetriever] Partial search: Processing {len(target_subqueries)} specific subqueries")

        # Use same adaptive batch processing logic as full search
        return await self.search_subqueries(target_subqueries, provider=provider)

    def _calculate_search_stats(self, results: Dict[str, SubqueryResult],
                               total_time: float) -> SearchStats:
        """Calculate comprehensive search statistics (same as original)"""
        if not results:
            return self._create_empty_stats()

        from collections import defaultdict

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
        """Create empty statistics object for error cases (same as original)"""
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

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory and performance statistics"""
        base_stats = {
            "retriever_searches": self.total_searches,
            "retriever_cache_hits": self.total_cache_hits,
            "retriever_cache_hit_rate": self.total_cache_hits / max(self.total_searches, 1),
            "retriever_avg_search_time": sum(self.search_times) / max(len(self.search_times), 1),
            "retriever_initialized": self.initialized
        }

        # Merge with manager memory stats
        manager_stats = self.manager.get_memory_stats()
        base_stats.update(manager_stats)

        return base_stats

    def _apply_temporal_filter(
        self,
        results: List[SearchResult],
        temporal_constraint: str
    ) -> List[SearchResult]:
        """
        Filter search results based on temporal constraint

        Args:
            results: List of SearchResult objects from FAISS search
            temporal_constraint: String like "EARLY (0-30%)", "MIDDLE", "LATE", "ALL"

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
            # Unknown constraint, don't filter
            logger.warning(f"[_apply_temporal_filter] Unknown temporal constraint: {temporal_constraint}")
            return results

        logger.info(f"[_apply_temporal_filter] Filtering with region [{region_start}, {region_end}] from constraint: {temporal_constraint}")

        # Filter results based on relative_position in metadata
        filtered = []
        filtered_out_count = 0
        for i, result in enumerate(results):
            # Get relative position from metadata
            rel_pos = result.metadata.get("relative_position")

            if rel_pos is None:
                # Fallback 1: estimate from line_number and total_lines
                line_num = result.metadata.get("line_number")
                total_lines = result.metadata.get("total_lines")

                if line_num is not None and total_lines is not None and total_lines > 0:
                    rel_pos = line_num / total_lines
                else:
                    # Fallback 2: estimate from Range (for old FAISS indexes)
                    range_info = result.metadata.get("Range")
                    if range_info and len(range_info) == 2:
                        line_num = (range_info[0] + range_info[1]) // 2
                        # Estimate total_lines from Range (assume Range represents similar proportion)
                        # This is less accurate but works for backward compatibility
                        total_lines = result.metadata.get("total_lines", 3000)
                        rel_pos = line_num / total_lines if total_lines > 0 else 0.5
                    else:
                        # Last resort: middle position
                        rel_pos = 0.5
                        logger.warning(f"[_apply_temporal_filter] No position metadata for result, using middle (0.5)")

            # Check if within temporal region
            if region_start <= rel_pos <= region_end:
                filtered.append(result)
                if i < 3:  # 첫 3개만 로깅
                    logger.debug(f"[_apply_temporal_filter] doc[{i}] PASS: rel_pos={rel_pos:.4f} in [{region_start}, {region_end}]")
            else:
                filtered_out_count += 1
                if i < 3:  # 첫 3개만 로깅
                    logger.debug(f"[_apply_temporal_filter] doc[{i}] REJECT: rel_pos={rel_pos:.4f} NOT in [{region_start}, {region_end}]")

        logger.info(f"[_apply_temporal_filter] Result: {len(filtered)} passed, {filtered_out_count} rejected")
        return filtered

# ─────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────

__all__ = [
    "MemoryOptimizedRetriever"
]