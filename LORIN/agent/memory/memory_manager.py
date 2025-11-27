"""
Memory-Optimized FAISS Manager - Singleton Pattern for GPU Memory Efficiency
=============================================================================

Prevents CUDA OOM by managing FAISS engine lifecycle and implementing aggressive
memory cleanup strategies for production retrieval operations.

Key Features:
- Singleton FAISS engine with lazy loading
- GPU memory monitoring and pressure relief
- Automatic cleanup on memory pressure
- Session-based memory tracking
- Safe fallback mechanisms

Architecture:
- MemoryOptimizedFAISSManager: Singleton controller
- GPUMemoryMonitor: Real-time memory tracking
- Memory pressure detection and cleanup automation
"""

from __future__ import annotations
import threading
import gc
import time
import torch
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from ...make_faiss.search_engine import FAISSSearchEngine, SearchResult
from ...logger.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# Memory Monitoring Data Classes
# ─────────────────────────────────────────────────────────

@dataclass
class GPUMemorySnapshot:
    """GPU memory state snapshot"""
    device_id: int
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float
    utilization_pct: float
    timestamp: float

@dataclass
class MemoryPressureLevel:
    """Memory pressure classification"""
    LOW = "low"        # < 60% usage
    MEDIUM = "medium"  # 60-80% usage
    HIGH = "high"      # 80-95% usage
    CRITICAL = "critical"  # > 95% usage

# ─────────────────────────────────────────────────────────
# GPU Memory Monitor
# ─────────────────────────────────────────────────────────

class GPUMemoryMonitor:
    """Real-time GPU memory monitoring and pressure detection"""

    def __init__(self, device_id: int = 0, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        self.device_id = device_id
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.snapshots: List[GPUMemorySnapshot] = []
        self.max_snapshots = 100

    def get_memory_snapshot(self) -> Optional[GPUMemorySnapshot]:
        """Get current GPU memory state"""
        if not torch.cuda.is_available():
            return None

        try:
            torch.cuda.synchronize(self.device_id)

            device_props = torch.cuda.get_device_properties(self.device_id)
            total_memory = device_props.total_memory
            allocated = torch.cuda.memory_allocated(self.device_id)
            reserved = torch.cuda.memory_reserved(self.device_id)

            total_gb = total_memory / (1024**3)
            allocated_gb = allocated / (1024**3)
            reserved_gb = reserved / (1024**3)
            free_gb = total_gb - allocated_gb
            utilization = allocated_gb / total_gb

            snapshot = GPUMemorySnapshot(
                device_id=self.device_id,
                allocated_gb=allocated_gb,
                reserved_gb=reserved_gb,
                free_gb=free_gb,
                total_gb=total_gb,
                utilization_pct=utilization * 100,
                timestamp=time.time()
            )

            # Store snapshot (rolling window)
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots.pop(0)

            return snapshot

        except Exception as e:
            logger.error(f"[GPUMemoryMonitor] Failed to get memory snapshot: {e}")
            return None

    def get_pressure_level(self) -> str:
        """Determine current memory pressure level"""
        snapshot = self.get_memory_snapshot()
        if not snapshot:
            return MemoryPressureLevel.LOW

        utilization = snapshot.utilization_pct / 100

        if utilization >= self.critical_threshold:
            return MemoryPressureLevel.CRITICAL
        elif utilization >= self.warning_threshold:
            return MemoryPressureLevel.HIGH
        elif utilization >= 0.6:
            return MemoryPressureLevel.MEDIUM
        else:
            return MemoryPressureLevel.LOW

    def force_cleanup(self) -> bool:
        """Force aggressive memory cleanup"""
        try:
            logger.info("[GPUMemoryMonitor] Forcing memory cleanup")

            # Clear PyTorch cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Trigger garbage collection
            gc.collect()

            # Wait a moment for cleanup to take effect
            time.sleep(0.1)

            return True

        except Exception as e:
            logger.error(f"[GPUMemoryMonitor] Force cleanup failed: {e}")
            return False

    def log_memory_status(self, context: str = ""):
        """Log current memory status"""
        snapshot = self.get_memory_snapshot()
        if snapshot:
            pressure = self.get_pressure_level()
            logger.info(
                f"[GPU Memory {context}] "
                f"Used: {snapshot.allocated_gb:.1f}GB/{snapshot.total_gb:.1f}GB "
                f"({snapshot.utilization_pct:.1f}%) | "
                f"Free: {snapshot.free_gb:.1f}GB | "
                f"Pressure: {pressure}"
            )

# ─────────────────────────────────────────────────────────
# Memory-Optimized FAISS Manager (Singleton)
# ─────────────────────────────────────────────────────────

class MemoryOptimizedFAISSManager:
    """
    Singleton FAISS engine manager with aggressive memory optimization

    Prevents CUDA OOM by:
    1. Reusing single FAISS engine instance
    2. Memory pressure monitoring
    3. Automatic cleanup on high pressure
    4. Lazy loading with cleanup triggers
    """

    _instance: Optional['MemoryOptimizedFAISSManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'MemoryOptimizedFAISSManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize singleton instance (only called once)"""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.search_engine: Optional[FAISSSearchEngine] = None
        self.memory_monitor = GPUMemoryMonitor()
        self.config: Optional[Dict[str, Any]] = None
        self.last_cleanup_time = 0.0
        self.cleanup_interval = 30.0  # Cleanup every 30 seconds under pressure
        self.search_count = 0
        self.total_search_time = 0.0

        logger.info("[MemoryOptimizedFAISSManager] Singleton instance created")

    def initialize_engine(self, config: Dict[str, Any]) -> bool:
        """
        Initialize FAISS engine with memory-safe configuration

        Args:
            config: Engine configuration including paths and parameters

        Returns:
            bool: True if initialization successful
        """
        try:
            # Check if already initialized with same config
            if self.search_engine is not None and self.config == config:
                logger.info("[MemoryOptimizedFAISSManager] Engine already initialized")
                return True

            # Memory pressure check before initialization
            pressure = self.memory_monitor.get_pressure_level()
            if pressure == MemoryPressureLevel.CRITICAL:
                logger.warning("[MemoryOptimizedFAISSManager] Critical memory pressure - forcing cleanup")
                self.memory_monitor.force_cleanup()

                # Recheck after cleanup
                pressure = self.memory_monitor.get_pressure_level()
                if pressure == MemoryPressureLevel.CRITICAL:
                    logger.error("[MemoryOptimizedFAISSManager] Insufficient memory after cleanup")
                    return False

            # Log memory before initialization
            self.memory_monitor.log_memory_status("Before Engine Init")

            # Clean up existing engine if configuration changed
            if self.search_engine is not None:
                logger.info("[MemoryOptimizedFAISSManager] Configuration changed - reinitializing engine")
                self._cleanup_engine()

            # Initialize new engine
            logger.info("[MemoryOptimizedFAISSManager] Initializing FAISS search engine")
            self.search_engine = FAISSSearchEngine(
                index_path=config.get("index_path"),
                model_name=config.get("model_name", "BAAI/bge-multilingual-gemma2"),
                device=config.get("device", "cuda"),
                use_fp16=config.get("use_fp16", True),
                max_length=config.get("max_length", 2048)
            )

            self.config = config.copy()

            # Log memory after initialization
            self.memory_monitor.log_memory_status("After Engine Init")

            logger.info("[MemoryOptimizedFAISSManager] Engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"[MemoryOptimizedFAISSManager] Engine initialization failed: {e}")
            self._cleanup_engine()
            return False

    def search(self, query: str, k: int = 5, score_threshold: Optional[float] = None,
               filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Memory-safe search with automatic cleanup

        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum score threshold
            filter: Optional metadata filter for pre-filtering (NEW)

        Returns:
            List[SearchResult]: Search results
        """
        if self.search_engine is None:
            logger.error("[MemoryOptimizedFAISSManager] Engine not initialized")
            return []

        search_start = time.time()

        try:
            # Memory pressure check before search
            pressure = self.memory_monitor.get_pressure_level()
            if pressure in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                logger.warning(f"[MemoryOptimizedFAISSManager] {pressure} memory pressure detected")
                self._conditional_cleanup()

            # Execute search with optional filter
            results = self.search_engine.search(
                query=query,
                k=k,
                score_threshold=score_threshold,
                filter=filter  # Pass filter to search engine
            )

            # Update search statistics
            search_time = time.time() - search_start
            self.search_count += 1
            self.total_search_time += search_time

            # Periodic cleanup based on memory pressure and time
            if self._should_cleanup():
                self._conditional_cleanup()

            logger.debug(f"[MemoryOptimizedFAISSManager] Search completed: {len(results)} results in {search_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"[MemoryOptimizedFAISSManager] Search failed: {e}")

            # If search fails due to memory, try cleanup and retry once
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.info("[MemoryOptimizedFAISSManager] Memory error detected - attempting recovery")
                self._emergency_cleanup()

                try:
                    # Single retry after cleanup
                    results = self.search_engine.search(query=query, k=k, score_threshold=score_threshold)
                    logger.info("[MemoryOptimizedFAISSManager] Recovery successful")
                    return results
                except Exception as retry_e:
                    logger.error(f"[MemoryOptimizedFAISSManager] Recovery failed: {retry_e}")

            return []

    def _should_cleanup(self) -> bool:
        """Determine if cleanup should be performed"""
        current_time = time.time()
        time_since_cleanup = current_time - self.last_cleanup_time

        # Cleanup triggers:
        # 1. High memory pressure
        pressure = self.memory_monitor.get_pressure_level()
        if pressure in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            return True

        # 2. Time-based cleanup under medium pressure
        if pressure == MemoryPressureLevel.MEDIUM and time_since_cleanup > self.cleanup_interval:
            return True

        # 3. Every 10 searches regardless of pressure
        if self.search_count % 10 == 0:
            return True

        return False

    def _conditional_cleanup(self):
        """Perform conditional memory cleanup"""
        pressure = self.memory_monitor.get_pressure_level()

        if pressure == MemoryPressureLevel.CRITICAL:
            self._emergency_cleanup()
        elif pressure == MemoryPressureLevel.HIGH:
            self._standard_cleanup()
        elif pressure == MemoryPressureLevel.MEDIUM:
            self._light_cleanup()

    def _light_cleanup(self):
        """Light memory cleanup - minimal performance impact"""
        try:
            torch.cuda.empty_cache()
            self.last_cleanup_time = time.time()
            logger.debug("[MemoryOptimizedFAISSManager] Light cleanup performed")
        except Exception as e:
            logger.error(f"[MemoryOptimizedFAISSManager] Light cleanup failed: {e}")

    def _standard_cleanup(self):
        """Standard memory cleanup"""
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            self.last_cleanup_time = time.time()
            logger.info("[MemoryOptimizedFAISSManager] Standard cleanup performed")
        except Exception as e:
            logger.error(f"[MemoryOptimizedFAISSManager] Standard cleanup failed: {e}")

    def _emergency_cleanup(self):
        """Emergency cleanup - aggressive memory recovery"""
        try:
            logger.warning("[MemoryOptimizedFAISSManager] Performing emergency cleanup")

            # Multiple cleanup passes
            for i in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.05)  # Brief pause between passes

            self.last_cleanup_time = time.time()
            self.memory_monitor.log_memory_status("After Emergency Cleanup")

        except Exception as e:
            logger.error(f"[MemoryOptimizedFAISSManager] Emergency cleanup failed: {e}")

    def _cleanup_engine(self):
        """Clean up the FAISS engine"""
        try:
            if self.search_engine is not None:
                logger.info("[MemoryOptimizedFAISSManager] Cleaning up FAISS engine")

                # Move models to CPU if possible
                if hasattr(self.search_engine, 'model') and self.search_engine.model is not None:
                    self.search_engine.model.cpu()

                # Clear references
                self.search_engine = None
                self.config = None

                # Force cleanup
                self._emergency_cleanup()

        except Exception as e:
            logger.error(f"[MemoryOptimizedFAISSManager] Engine cleanup failed: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        snapshot = self.memory_monitor.get_memory_snapshot()
        if not snapshot:
            return {}

        return {
            "gpu_allocated_gb": snapshot.allocated_gb,
            "gpu_free_gb": snapshot.free_gb,
            "gpu_utilization_pct": snapshot.utilization_pct,
            "pressure_level": self.memory_monitor.get_pressure_level(),
            "search_count": self.search_count,
            "avg_search_time": self.total_search_time / max(self.search_count, 1),
            "engine_initialized": self.search_engine is not None
        }

    def shutdown(self):
        """Shutdown and cleanup manager"""
        logger.info("[MemoryOptimizedFAISSManager] Shutting down")
        self._cleanup_engine()

    @classmethod
    def reset_singleton(cls):
        """Reset singleton instance (for testing)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

# ─────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────

__all__ = [
    "MemoryOptimizedFAISSManager",
    "GPUMemoryMonitor",
    "GPUMemorySnapshot",
    "MemoryPressureLevel"
]