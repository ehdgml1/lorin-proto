"""
LORIN Agent Module - LangGraph-based Multi-Agent System

Core module providing an advanced multi-agent system based on LangGraph.
Models complex AI workflows as state graphs for processing.

Module Structure:
    agent/
    ├── state.py            # LangGraph state management
    ├── schema.py           # Metadata schema and TypedDict
    ├── graph.py            # LangGraph StateGraph utilities
    │
    ├── planning/           # Query decomposition and planning
    │   ├── planner.py
    │   ├── replanner.py
    │   ├── intent_classifier.py
    │   └── question_decomposer.py
    │
    ├── retrieval/          # Document retrieval
    │   ├── retrieve.py
    │   ├── memory_optimized_retrieve_node.py
    │   └── memory_optimized_retriever.py
    │
    ├── evaluation/         # Quality evaluation
    │   ├── quality_evaluator.py
    │   └── quality_evaluator_node.py
    │
    ├── routing/            # Routing and loop prevention
    │   ├── routing_decision.py
    │   └── loop_prevention.py
    │
    ├── memory/             # Memory management
    │   └── memory_manager.py
    │
    └── generation/         # Answer generation
        └── answer.py

Backward Compatibility:
    All existing import paths are maintained:
        from LORIN.agent import planning_node
        from LORIN.agent import retrieve_node
        from LORIN.agent import quality_evaluator_node
        etc...
"""

# Foundation (root level)
from .state import AgentState, create_initial_state, format_agent_state
from .schema import (
    MetadataManager,
    LoopPreventionData,
    LoopDetectionResult,
    PatternDetectionResult,
    RoutingDecision,
    LastRoutingDecision,
    FailedQueryInfo,
    IterationHistoryEntry,
    QueryEvaluationInfo,
    LoopPreventionStatus,
    create_replanner_metadata,
    get_subqueries,
    create_answer_metadata,
    get_search_results,
    get_quality_filtered_search_results,
)
from .graph import create_agent_graph, llm_node

# Planning Module
from .planning.planner import planning_node
from .planning.replanner import replanning_node
from .planning.intent_classifier import determine_intent, classify_intent_with_llm
from .planning.question_decomposer import decompose_question

# Retrieval Module
from .retrieval.retrieve import (
    retrieve_node,
    FAISSRetriever,
    SubqueryResult,
    SearchStats,
    RetrievalConfig,
    get_retrieval_results_for_query,
)
from .retrieval.memory_optimized_retrieve_node import (
    retrieve_node as memory_optimized_retrieve_node,
)
from .retrieval.memory_optimized_retriever import MemoryOptimizedRetriever

# Evaluation Module
from .evaluation.quality_evaluator import (
    QualityEvaluator,
    QueryEvaluationResult,
    DocumentEvaluation,
    EvaluationStats,
)
from .evaluation.quality_evaluator_node import quality_evaluator_node

# Routing Module
from .routing.routing_decision import (
    determine_iterative_routing,
    determine_iterative_routing_with_loop_prevention,
    should_retry_routing,
    analyze_query_results,
    identify_persistent_failures,
)
from .routing.loop_prevention import (
    initialize_loop_prevention,
    detect_routing_loop,
    detect_oscillation_pattern,
    detect_consecutive_pattern,
    detect_cycle_pattern,
    log_forced_termination,
    update_loop_prevention_metadata,
)

# Memory Module
from .memory.memory_manager import MemoryOptimizedFAISSManager

# Generation Module
from .generation.answer import answer_node

# Public API
__all__ = [
    # Foundation
    "AgentState",
    "create_initial_state",
    "format_agent_state",
    "MetadataManager",
    "create_agent_graph",
    "llm_node",

    # Planning
    "planning_node",
    "replanning_node",
    "determine_intent",
    "classify_intent_with_llm",
    "decompose_question",

    # Retrieval
    "retrieve_node",
    "memory_optimized_retrieve_node",
    "FAISSRetriever",
    "MemoryOptimizedRetriever",
    "SubqueryResult",
    "SearchStats",
    "RetrievalConfig",
    "get_retrieval_results_for_query",

    # Evaluation
    "QualityEvaluator",
    "quality_evaluator_node",
    "QueryEvaluationResult",
    "DocumentEvaluation",
    "EvaluationStats",

    # Routing
    "determine_iterative_routing",
    "determine_iterative_routing_with_loop_prevention",
    "should_retry_routing",
    "analyze_query_results",
    "identify_persistent_failures",
    "initialize_loop_prevention",
    "detect_routing_loop",

    # Memory
    "MemoryOptimizedFAISSManager",

    # Generation
    "answer_node",

    # Schema types
    "LoopPreventionData",
    "LoopDetectionResult",
    "RoutingDecision",
    "FailedQueryInfo",
    "QueryEvaluationInfo",
    "LoopPreventionStatus",
]

__version__ = "0.2.0"
