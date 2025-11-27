"""
LORIN - Log Observation and Reasoning Intelligence Network
---

LORIN is a LangGraph-based multi-agent log analysis system.
Supports various LLM models (Gemini, Claude, GPT, EXAONE) with
agent-based log analysis workflows and session management.

Module Structure
---
- **agent**: LangGraph-based agent system
- **llm**: Unified LLM interface and session management
- **logger**: Advanced logging and monitoring
- **process**: Data processing and workflow management
- **prompt**: Prompt template and management

Author: CAU HILab & BALab
Version: 0.1.0
License: None
"""

__version__ = "0.1.0"
__author__ = "LORIN Team"
__email__ = "cau.caw.chat@cau.ac.kr"
__license__ = "No License"

# Main component imports
from .agent import (
    AgentState,
    create_initial_state,
    format_agent_state,
    create_agent_graph,
    llm_node
)

from .llm import (
    Chatbot,
    LLMProvider,
    TokenCounter,
    ModelLimits,
    analyze_text_tokens
)

from .logger import (
    Logger,
    get_logger,
    LogLevel
)

from .process import (
    main_process,
    initialize_graph
)

# Package-level exports
__all__ = [
    # Graph related
    "AgentState",
    "create_initial_state",
    "format_agent_state",
    "create_agent_graph",
    "llm_node",

    # LLM related
    "Chatbot",
    "LLMProvider",
    "TokenCounter",
    "ModelLimits",
    "analyze_text_tokens",

    # Logger related
    "Logger",
    "get_logger",
    "LogLevel",

    # Process related
    "main_process",
    "initialize_graph"
]
