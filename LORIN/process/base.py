"""
Process Base Module - Data Processing and Workflow Management

This module integrates Agents and LLMs to manage complex data processing workflows.
Supports various processing patterns with an extensible architecture.
"""

from ..logger.logger import get_logger
from ..llm import Chatbot
from ..agent.graph import create_agent_graph
from ..agent.state import create_initial_state, get_last_message, format_agent_state
from ..process.route import initialize_graph

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime, timezone

from typing import Optional


async def _single_turn_conversation(app: StateGraph, questions: list[str], config: Optional[dict] = None) -> str:
    """
    Execute single-turn conversation where each question creates a fresh state.
    Returns a list of final states for each question.
    """
    logger = get_logger(__name__)
    logger.debug("Single-turn conversation begins")

    states = []

    for i, question in enumerate(questions, 1):
        metadata = {
            "question_id": i,
            "experiment_config": config or {}
        }
        current_state = create_initial_state(
            question,
            metadata=metadata
        )

        logger.info(f"[Q{i}] {question}")

        try:
            current_state = await app.ainvoke(current_state)

            last_message = get_last_message(current_state)
            logger.info(f"[A{i}] {last_message.content}")

            states.append(current_state)

        except Exception as e:
            logger.error(f"Question {i} processing failed: {e}")

    logger.debug("Single-turn conversation ends")

    return states


async def _multi_turn_conversation(app: StateGraph, questions: list[str], config: Optional[dict] = None) -> str:
    """
    Execute multi-turn conversation where all questions share a single state.
    Returns the final accumulated state.
    """
    logger = get_logger(__name__)
    logger.debug("Multi-turn conversation begins")

    metadata = {
        "experiment_config": config or {}
    }
    current_state = create_initial_state(
        questions[0],
        metadata=metadata
    )

    for i, question in enumerate(questions, 1):
        logger.info(f"[Q{i}] {question}")

        if i > 1:
            current_state["messages"].append(
                HumanMessage(
                    content=question,
                    kwargs={
                        "agent_name": "human",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

        try:
            current_state = await app.ainvoke(current_state)

            last_message = get_last_message(current_state)
            logger.info(f"[A{i}] {last_message.content}")

        except Exception as e:
            logger.error(f"Question {i} processing failed: {e}")

    logger.debug("Multi-turn conversation ends")

    return current_state


async def _llm_process(app: StateGraph, config: Optional[dict] = None, question: Optional[str] = None):
    """
    Main LLM processing function that handles question preparation and answer generation.
    Supports both experiment mode (external questions) and development mode (hardcoded questions).
    """
    logger = get_logger(__name__)

    logger.debug("LLM process begins")

    if question:
        questions = [question]
        logger.info(f"Using provided question: {question[:100]}...")
    else:
        questions = [
            "The data passed between processes is too large and this is causing a failure. Please tell me which log range I should check for debugging."
        ]
        logger.info("Using default hardcoded question (development mode)")

    conversation_type = "single"

    if conversation_type == "single":
        states = await _single_turn_conversation(app, questions, config)
    elif conversation_type == "multi":
        state = await _multi_turn_conversation(app, questions, config)
        states = [state]
    else:
        logger.warning("Invalid conversation type")
        states = await _single_turn_conversation(app, questions, config)

    for state in states:
        logger.info(f'{format_agent_state(state)}')

    logger.debug("LLM process ends")

    return states


async def main_process(
    chatbot: Chatbot,
    vectorstore,
    *,
    sparse_store=None,
    corpus_path: Optional[str] = None,
    config: Optional[dict] = None,
    question: Optional[str] = None
):
    """
    Main entry point for the processing pipeline.
    Creates and initializes the agent graph, then executes the LLM process.
    """
    logger = get_logger(__name__)
    logger.debug("Main process begins")

    if config:
        logger.info(f"Main process running with config: {config}")
    if question:
        logger.info(f"Main process running with question: {question[:100]}...")

    graph = create_agent_graph()
    logger.debug("1.Gragh created")

    graph = await initialize_graph(
        graph,
        chatbot,
        vectorstore,
        corpus_path=corpus_path,
        sparse_store=sparse_store,
        config=config
    )
    logger.debug("2.Graph initialized")

    app = graph.compile()
    logger.debug("3.Graph compiled")

    states = await _llm_process(app, config, question)
    logger.debug("4.LLM process ends")
    return states
