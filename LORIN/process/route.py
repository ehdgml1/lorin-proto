# LORIN/process/route.py
"""
Process Route Module - LangGraph and Route Management with Feedback Loop

Manages the main LangGraph process graph with feedback loop support.
Optimized for BGE retrieval.
"""

from typing import Optional
from ..logger.logger import get_logger
from ..llm import Chatbot
# Using re-exported public API after subfolder restructuring
from ..agent import (
    planning_node,
    replanning_node,
    answer_node,
    quality_evaluator_node,
    should_retry_routing,
    memory_optimized_retrieve_node as retrieve_node,
)
from ..agent.planning.planner import pivot_transition_node  # Sequential workflow support
from langgraph.graph import StateGraph, START, END


# --- Sequential Workflow Routing Functions ---

def should_expand(state) -> bool:
    """Determine if pivot extraction and expansion is needed (Sequential workflow)

    This function checks if we're in the initial phase of sequential evidence expansion
    and need to transition to the expansion phase via pivot extraction.

    Args:
        state: Current agent state

    Returns:
        bool: True if in initial phase and needs expansion, False otherwise
    """
    planner_data = state.get("metadata", {}).get("planner", {})
    phase = planner_data.get("phase", "parallel")
    needs_expansion = planner_data.get("needs_expansion", False)

    logger = get_logger(__name__)

    # Only expand if in initial phase and expansion is needed
    if phase == "initial" and needs_expansion:
        logger.info("[routing] Sequential mode detected: Transitioning to pivot extraction")
        return True
    else:
        logger.debug(f"[routing] Continuing normal flow (phase={phase}, needs_expansion={needs_expansion})")
        return False


# --- Node Addition ---

async def _add_nodes(
    graph: StateGraph,
    chatbot: Chatbot,
    vectorstore,
) -> StateGraph:
    """Add nodes to the main graph.

    Includes feedback loop support. Only adds nodes defined in LORIN.agent.* modules.
    Creates independent instances per node for parallel processing support.
    """
    logger = get_logger(__name__)
    logger.debug("Add nodes begins (with independent instances & parallel processing)")

    if vectorstore is None:
        raise ValueError("vectorstore is required to initialize the retrieve node.")

    # Create independent Chatbot instances per node
    # Each node uses independent instances for parallel execution:
    # - EXAONE (local): ExaoneWrapper singleton shares internal model
    # - Gemini/GPT/Llama (API): Parallel API calls supported

    from ..utils import create_chatbot_from_env

    logger.info("Creating independent chatbot instances for each node...")

    # 1) Planner Bot - plan generation (balanced creativity)
    planner_bot = create_chatbot_from_env(
        temperature=0.4,
        max_tokens=32768
    )
    logger.info(f"  Planner bot: {planner_bot.provider.value}/{planner_bot.model} (temp=0.4, max_tokens=32768)")

    # 2) Quality Evaluator Bot - evaluation consistency (low temperature)
    evaluator_bot = create_chatbot_from_env(
        temperature=0.1,
        max_tokens=32768
    )
    logger.info(f"  Evaluator bot: {evaluator_bot.provider.value}/{evaluator_bot.model} (temp=0.1, max_tokens=32768)")

    # 3) Replanner Bot - replan generation (balanced creativity)
    replanner_bot = create_chatbot_from_env(
        temperature=0.4,
        max_tokens=32768
    )
    logger.info(f"  Replanner bot: {replanner_bot.provider.value}/{replanner_bot.model} (temp=0.4, max_tokens=32768)")

    # 4) Answer Bot - final answer (consistent output)
    answer_bot = create_chatbot_from_env(
        temperature=0.1,
        max_tokens=32768
    )
    logger.info(f"  Answer bot: {answer_bot.provider.value}/{answer_bot.model} (temp=0.1, max_tokens=32768)")

    # Add nodes (each with independent instance)

    # 1) Planner node
    graph.add_node(
        "planner",
        planning_node(
            chatbot=planner_bot,
            agent_name="planner",
        )
    )

    # 2) Memory-optimized retriever node (parallel subquery processing)
    graph.add_node(
        "retrieve",
        retrieve_node
    )

    # 3) Quality evaluator node
    graph.add_node(
        "quality_evaluator",
        quality_evaluator_node(chatbot=evaluator_bot)
    )

    # 4) Replanner node
    graph.add_node(
        "replanner",
        await replanning_node(
            chatbot=replanner_bot,
            agent_name="replanner",
        )
    )

    # 5) Answer node
    graph.add_node(
        "answer",
        answer_node(
            answer_llm=answer_bot,
            agent_name="answer",
        )
    )

    # 6) Pivot transition node (for sequential evidence expansion workflow)
    graph.add_node(
        "pivot_transition",
        pivot_transition_node()
    )

    logger.debug("Add nodes ends (6 nodes with independent instances)")
    logger.info("All nodes configured with parallel processing and sequential expansion support")
    return graph


# --- Edge Addition ---

def _add_edges(graph: StateGraph, config: Optional[dict] = None) -> StateGraph:
    """Add edges to the main graph.

    Includes iterative quality improvement and sequential expansion support.

    Flows:
    - Parallel (default): START -> planner -> retrieve -> quality_evaluator -> [replanner|retrieve|answer] -> END
    - Sequential: START -> planner(initial) -> retrieve -> pivot_transition -> planner(expansion) -> retrieve -> quality_evaluator -> [replanner|retrieve|answer] -> END
    - Iteration: replanner -> retrieve -> quality_evaluator -> ... (max 5 iterations)

    Ablation Support:
    - wo_Planner: START -> retrieve -> quality_evaluator -> answer
    - wo_QualityEvaluator: START -> planner -> retrieve -> [should_expand] -> {pivot_transition -> planner | answer} -> END
    - wo_Replanner: START -> planner -> retrieve -> [should_expand] -> {pivot_transition -> planner | quality_evaluator} -> answer -> END
    - LORIN_Full (default): All components enabled (sequential expansion + replanning loop)
    """
    logger = get_logger(__name__)
    logger.debug("Add edges begins (iterative quality improvement + sequential expansion)")

    # Extract ablation flags
    skip_planner = config.get("skip_planner", False) if config else False
    skip_quality_evaluator = config.get("skip_quality_evaluator", False) if config else False
    skip_replanner = config.get("skip_replanner", False) if config else False

    logger.info(f"[_add_edges] Ablation config: skip_planner={skip_planner}, "
                f"skip_quality_evaluator={skip_quality_evaluator}, skip_replanner={skip_replanner}")

    # 1. wo_Planner: START -> retrieve -> quality_evaluator -> answer
    if skip_planner:
        logger.info("[_add_edges] Ablation: wo_Planner - Bypassing planner node")
        graph.add_edge(START, "retrieve")

        if skip_quality_evaluator:
            # wo_Planner + wo_QualityEvaluator: START -> retrieve -> answer
            logger.info("[_add_edges] Ablation: wo_Planner + wo_QualityEvaluator")
            graph.add_edge("retrieve", "answer")
        else:
            # wo_Planner only: START -> retrieve -> quality_evaluator -> answer
            graph.add_edge("retrieve", "quality_evaluator")
            graph.add_edge("quality_evaluator", "answer")

        graph.add_edge("answer", END)
        logger.info("wo_Planner graph configuration complete")
        return graph

    # 2. wo_QualityEvaluator: START -> planner -> retrieve -> [should_expand] -> {pivot_transition | answer}
    if skip_quality_evaluator:
        logger.info("[_add_edges] Ablation: wo_QualityEvaluator - Bypassing quality evaluator node")
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retrieve")

        # Sequential expansion support: Check if pivot extraction is needed
        graph.add_conditional_edges(
            "retrieve",
            should_expand,
            {
                True: "pivot_transition",      # Sequential mode: Extract pivot and transition to expansion
                False: "answer"                # Parallel mode: Direct to answer (bypassing quality evaluator)
            }
        )

        # Sequential workflow: After pivot extraction, go back to planner for expansion queries
        graph.add_edge("pivot_transition", "planner")

        graph.add_edge("answer", END)
        logger.info("wo_QualityEvaluator graph configuration complete (with sequential expansion support)")
        return graph

    # 3. LORIN_Full or wo_Replanner: Normal flow with conditional replanner

    # Default flow
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieve")

    # Conditional routing after retrieve: Check if sequential expansion is needed
    graph.add_conditional_edges(
        "retrieve",
        should_expand,
        {
            True: "pivot_transition",      # Sequential mode: Extract pivot and transition to expansion
            False: "quality_evaluator"     # Parallel mode: Continue to quality evaluation (default)
        }
    )

    # Sequential workflow: After pivot extraction, go back to planner for expansion queries
    graph.add_edge("pivot_transition", "planner")

    # Conditional routing - decide retry or answer after quality evaluation
    if skip_replanner:
        # wo_Replanner: quality_evaluator always routes to answer (no replanning)
        logger.info("[_add_edges] Ablation: wo_Replanner - Quality evaluator routes directly to answer")
        graph.add_edge("quality_evaluator", "answer")
    else:
        # LORIN_Full: Quality evaluator routes to replanner/retrieve/answer based on evaluation
        graph.add_conditional_edges(
            "quality_evaluator",
            should_retry_routing,  # routing function
            {
                "replanner": "replanner",  # replanning needed
                "retrieve": "retrieve",    # direct re-retrieval
                "answer": "answer"         # proceed to answer generation
            }
        )

        # After replanning, complete search -> quality evaluation cycle
        graph.add_edge("replanner", "retrieve")

    # Terminate after final answer
    graph.add_edge("answer", END)

    logger.debug("Add edges ends (iterative quality improvement + sequential expansion configured)")
    logger.info("Graph routing supports both parallel and sequential evidence expansion modes")
    return graph


# --- Graph Initialization ---

async def initialize_graph(
    graph: StateGraph,
    chatbot: Chatbot,
    vectorstore,
    *,
    corpus_path: Optional[str] = None,    # kept for compatibility (unused)  # pylint: disable=unused-argument
    sparse_store=None,                     # kept for compatibility (unused)  # pylint: disable=unused-argument
    config: Optional[dict] = None          # experiment config (Ablation, etc.)
) -> StateGraph:
    """Main process method for selective subquery reconstruction + sequential evidence expansion.

    Flows:
    - Parallel: START -> planner -> retrieve -> quality_evaluator -> [replanner|retrieve|answer] -> END
    - Sequential: START -> planner(initial) -> retrieve -> pivot_transition -> planner(expansion) -> retrieve -> quality_evaluator -> [replanner|retrieve|answer] -> END
    - Iteration: Only failed subqueries are reconstructed to progressively improve quality

    Args:
        graph: LangGraph StateGraph instance
        chatbot: Chatbot instance
        vectorstore: FAISS vector store
        corpus_path: Kept for compatibility (unused)
        sparse_store: Kept for compatibility (unused)
        config: Experiment settings (optional)
            Example: {"use_planner": True, "use_activity_detection": False, "sequential_mode": True, ...}
            Ablation: {"skip_planner": True/False, "skip_quality_evaluator": True/False, "skip_replanner": True/False}

    Mode Selection:
        - Parallel (default): When metadata["planner"]["phase"] is "parallel" or not set
        - Sequential: Automatically activated when metadata["planner"]["phase"] = "initial"
    """
    logger = get_logger(__name__)
    logger.debug("Initialize graph begins (selective subquery reconstruction system)")

    if config:
        logger.info(f"Graph initialization with config: {config}")

    # LORIN Flow
    graph = await _add_nodes(graph, chatbot, vectorstore)
    logger.debug("1. Nodes added (6 nodes): planner, retrieve, quality_evaluator, replanner, answer, pivot_transition")

    graph = _add_edges(graph, config=config)
    logger.debug("2. Edges added (iterative quality improvement + sequential expansion routing + ablation support)")

    logger.debug("Initialize graph ends (selective reconstruction + sequential expansion enabled)")
    logger.info("Graph supports parallel (default) and sequential evidence expansion modes")
    return graph
