"""
LORIN Centralized Settings
---

Centralized management of all environment variables and settings
for improved type safety and maintainability
"""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class LLMSettings:
    """LLM settings

    Supported EXAONE models:
    - "exaone-3.5-7.8b" or "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    - "exaone-4.0.1-32b" or "LGAI-EXAONE/EXAONE-4.0.1-32B" (default)
    """
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "exaone"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "exaone-4.0.1-32b"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.4")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048")))

    # Retry settings (optimized)
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "30.0")))


@dataclass
class RetrievalSettings:
    """Retrieval settings"""
    # FAISS index
    index_path: str = field(default_factory=lambda: os.getenv(
        "FAISS_INDEX_PATH",
        "LORIN/make_faiss/log_faiss_index_bge_gemma2"
    ))
    model_name: str = field(default_factory=lambda: os.getenv(
        "BGE_MODEL_NAME",
        "BAAI/bge-multilingual-gemma2"
    ))
    device: str = field(default_factory=lambda: os.getenv("BGE_DEVICE", "cuda"))

    # Search parameters
    top_k_default: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_TOP_K", "5")))
    top_k_max: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_TOP_K_MAX", "20")))
    query_max_length: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_QUERY_MAXLEN", "200")))

    # Quality thresholds
    success_threshold: float = field(default_factory=lambda: float(os.getenv("RETRIEVER_SUCCESS_THRESHOLD", "0.5")))
    quality_threshold: float = field(default_factory=lambda: float(os.getenv("RETRIEVER_QUALITY_THRESHOLD", "0.25")))

    # Caching settings (optimized)
    cache_max_size: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_CACHE_SIZE", "1000")))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_CACHE_TTL", "3600")))
    deduplication_enabled: bool = field(default_factory=lambda: os.getenv("RETRIEVER_DEDUP", "1") == "1")

    # Feature Flags for Retrieval Improvements (Phase 1 & 2)
    # Enable keyword extraction in expansion prompt (Expected improvement: +30-40%)
    use_keyword_extraction: bool = field(default_factory=lambda: os.getenv("USE_KEYWORD_EXTRACTION", "1") == "1")

    # Enable pre-filtering (search within position range) instead of post-filtering (Expected improvement: +60-70%)
    use_pre_filtering: bool = field(default_factory=lambda: os.getenv("USE_PRE_FILTERING", "1") == "1")


@dataclass
class PlannerSettings:
    """Planner settings"""
    subquery_min: int = field(default_factory=lambda: int(os.getenv("PLANNER_SUBQ_MIN", "1")))
    subquery_max: int = field(default_factory=lambda: int(os.getenv("PLANNER_SUBQ_MAX", "5")))
    force_anchor: bool = field(default_factory=lambda: os.getenv("PLANNER_FORCE_ANCHOR", "1") == "1")
    language: str = field(default_factory=lambda: os.getenv("PLANNER_LANGUAGE", "ko"))

    assumptions_line: str = field(default_factory=lambda: os.getenv(
        "ASSUMPTIONS_LINE",
        "Environment: Android OS logcat logs (only component + message available) "
        "Purpose: Query decomposition to find relevant logs for debugging user question issues"
    ))


@dataclass
class QualityEvaluatorSettings:
    """Quality Evaluator settings"""
    relevance_threshold: float = field(default_factory=lambda: float(os.getenv("QE_RELEVANCE_THRESHOLD", "0.5")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("QE_CONFIDENCE_THRESHOLD", "0.6")))
    max_docs_per_query: int = field(default_factory=lambda: int(os.getenv("QE_MAX_DOCS", "5")))

    # Optimized retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv("QE_MAX_RETRIES", "2")))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("QE_TIMEOUT", "30.0")))


@dataclass
class ReplannerSettings:
    """Replanner settings"""
    max_iterations: int = field(default_factory=lambda: int(os.getenv("REPLANNER_MAX_ITER", "25")))
    min_improvement_threshold: float = field(default_factory=lambda: float(os.getenv("REPLANNER_MIN_IMPROVE", "0.01")))

    # Optimized retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv("REPLANNER_MAX_RETRIES", "2")))

    # Per-subquery failure limit: skip subquery after N consecutive failures
    max_failures_per_subquery: int = field(default_factory=lambda: int(os.getenv("REPLANNER_MAX_FAILURES_PER_SUBQUERY", "5")))


@dataclass
class LoggerSettings:
    """Logging settings"""
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))

    # Optimization: structured logging
    structured_logging: bool = field(default_factory=lambda: os.getenv("STRUCTURED_LOGGING", "0") == "1")


@dataclass
class FAISSIndexSettings:
    """FAISS indexing settings

    Chunking and embedding parameters used when building FAISS vector index
    """
    # Chunking parameters
    window_size: int = field(default_factory=lambda: int(os.getenv("FAISS_WINDOW_SIZE", "100")))
    stride: int = field(default_factory=lambda: int(os.getenv("FAISS_STRIDE", "50")))
    min_len: int = field(default_factory=lambda: int(os.getenv("FAISS_MIN_LEN", "3")))

    # Embedding parameters
    max_token_length: int = field(default_factory=lambda: int(os.getenv("FAISS_MAX_TOKEN_LENGTH", "8192")))
    batch_size_per_gpu: int = field(default_factory=lambda: int(os.getenv("FAISS_BATCH_SIZE_PER_GPU", "32")))
    use_fp16: bool = field(default_factory=lambda: os.getenv("FAISS_USE_FP16", "1") == "1")
    use_advanced_multi_gpu: bool = field(default_factory=lambda: os.getenv("FAISS_ADVANCED_MULTI_GPU", "0") == "1")


@dataclass
class Stage1FilteringSettings:
    """Stage 1 anomaly detection settings

    Teacher-Student MAE based anomaly detection module parameters
    """
    # Window parameters
    window_size: int = field(default_factory=lambda: int(os.getenv("STAGE1_WINDOW_SIZE", "50")))
    stride: int = field(default_factory=lambda: int(os.getenv("STAGE1_STRIDE", "40")))

    # Training/inference parameters
    batch_size: int = field(default_factory=lambda: int(os.getenv("STAGE1_BATCH_SIZE", "64")))
    train_ratio: float = field(default_factory=lambda: float(os.getenv("STAGE1_TRAIN_RATIO", "0.9")))
    workers: int = field(default_factory=lambda: int(os.getenv("STAGE1_WORKERS", "4")))

    # Anomaly detection threshold
    anomaly_threshold: float = field(default_factory=lambda: float(os.getenv("STAGE1_ANOMALY_THRESHOLD", "0.3")))

    # Model checkpoint paths
    teacher_checkpoint: str = field(default_factory=lambda: os.getenv(
        "STAGE1_TEACHER_CKPT", "stage1_filtering/teacher_pretrained_aosp.pt"
    ))
    student_checkpoint_dir: str = field(default_factory=lambda: os.getenv(
        "STAGE1_STUDENT_CKPT_DIR", "stage1_filtering/revkd_out_v5_aosp"
    ))


@dataclass
class DrainParserSettings:
    """Drain log parser settings

    Drain algorithm parameters for log template extraction
    """
    depth: int = field(default_factory=lambda: int(os.getenv("DRAIN_DEPTH", "5")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("DRAIN_SIMILARITY_THRESHOLD", "0.5")))
    max_children: int = field(default_factory=lambda: int(os.getenv("DRAIN_MAX_CHILDREN", "100")))
    default_threshold: int = field(default_factory=lambda: int(os.getenv("DRAIN_DEFAULT_THRESHOLD", "5")))


@dataclass
class ExtendedLoggerSettings:
    """Extended logger settings

    File rotation and backup settings
    """
    max_bytes: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_BYTES", "10485760")))  # 10MB
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))


@dataclass
class SystemSettings:
    """System-wide settings"""
    # Debug mode
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "0") == "1")

    # Performance monitoring
    performance_monitoring: bool = field(default_factory=lambda: os.getenv("PERF_MONITORING", "1") == "1")

    # LangSmith settings
    langsmith_enabled: bool = field(default_factory=lambda: os.getenv("LANGSMITH_TRACING", "0") == "1")
    langsmith_project: Optional[str] = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT"))


# ---
# Global Settings Instance
# ---

@dataclass
class Settings:
    """Global settings class"""
    llm: LLMSettings = field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    planner: PlannerSettings = field(default_factory=PlannerSettings)
    quality_evaluator: QualityEvaluatorSettings = field(default_factory=QualityEvaluatorSettings)
    replanner: ReplannerSettings = field(default_factory=ReplannerSettings)
    logger: LoggerSettings = field(default_factory=LoggerSettings)
    system: SystemSettings = field(default_factory=SystemSettings)

    # Newly added settings
    faiss_index: FAISSIndexSettings = field(default_factory=FAISSIndexSettings)
    stage1_filtering: Stage1FilteringSettings = field(default_factory=Stage1FilteringSettings)
    drain_parser: DrainParserSettings = field(default_factory=DrainParserSettings)
    extended_logger: ExtendedLoggerSettings = field(default_factory=ExtendedLoggerSettings)

    def validate(self) -> bool:
        """Validate settings"""
        # LLM settings validation
        assert 0.0 <= self.llm.temperature <= 2.0, "Temperature must be between 0.0 and 2.0"
        assert self.llm.max_tokens > 0, "Max tokens must be positive"
        assert self.llm.max_retries >= 1, "Max retries must be at least 1"

        # Retrieval settings validation
        assert self.retrieval.top_k_default > 0, "Top-k must be positive"
        assert self.retrieval.cache_max_size > 0, "Cache size must be positive"
        assert self.retrieval.cache_ttl_seconds > 0, "Cache TTL must be positive"

        # Planner settings validation
        assert self.planner.subquery_min >= 1, "Subquery min must be at least 1"
        assert self.planner.subquery_max >= self.planner.subquery_min, "Subquery max must be >= min"

        # Quality Evaluator settings validation
        assert 0.0 <= self.quality_evaluator.relevance_threshold <= 1.0, "Threshold must be between 0 and 1"
        assert self.quality_evaluator.max_docs_per_query > 0, "Max docs must be positive"

        # FAISS index settings validation
        assert self.faiss_index.window_size > 0, "FAISS window size must be positive"
        assert self.faiss_index.stride > 0, "FAISS stride must be positive"
        assert self.faiss_index.max_token_length > 0, "FAISS max token length must be positive"

        # Stage 1 settings validation
        assert 0.0 <= self.stage1_filtering.anomaly_threshold <= 1.0, "Anomaly threshold must be between 0 and 1"
        assert self.stage1_filtering.workers >= 1, "Workers must be at least 1"

        # Drain parser settings validation
        assert self.drain_parser.depth >= 1, "Drain depth must be at least 1"
        assert 0.0 <= self.drain_parser.similarity_threshold <= 1.0, "Similarity threshold must be between 0 and 1"

        return True


# ---
# Singleton Instance
# ---

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)

    Returns:
        Settings: Validated global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings (when environment variables change)

    Returns:
        Settings: New settings instance
    """
    global _settings
    _settings = Settings()
    _settings.validate()
    return _settings


# ---
# Convenience Functions
# ---

def get_llm_settings() -> LLMSettings:
    """Get LLM settings only"""
    return get_settings().llm


def get_retrieval_settings() -> RetrievalSettings:
    """Get retrieval settings only"""
    return get_settings().retrieval


def get_planner_settings() -> PlannerSettings:
    """Get planner settings only"""
    return get_settings().planner


def get_faiss_settings() -> FAISSIndexSettings:
    """Get FAISS index settings only"""
    return get_settings().faiss_index


def get_stage1_settings() -> Stage1FilteringSettings:
    """Get Stage 1 anomaly detection settings only"""
    return get_settings().stage1_filtering


def get_drain_settings() -> DrainParserSettings:
    """Get Drain parser settings only"""
    return get_settings().drain_parser


def get_extended_logger_settings() -> ExtendedLoggerSettings:
    """Get extended logger settings only"""
    return get_settings().extended_logger


def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return get_settings().system.debug_mode


# ---
# Module Exports
# ---

__all__ = [
    "Settings",
    "LLMSettings",
    "RetrievalSettings",
    "PlannerSettings",
    "QualityEvaluatorSettings",
    "ReplannerSettings",
    "LoggerSettings",
    "SystemSettings",
    "FAISSIndexSettings",
    "Stage1FilteringSettings",
    "DrainParserSettings",
    "ExtendedLoggerSettings",
    "get_settings",
    "reload_settings",
    "get_llm_settings",
    "get_retrieval_settings",
    "get_planner_settings",
    "get_faiss_settings",
    "get_stage1_settings",
    "get_drain_settings",
    "get_extended_logger_settings",
    "is_debug_mode"
]
