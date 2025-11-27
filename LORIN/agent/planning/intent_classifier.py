"""
Intent Classifier Module - LLM-based Query Intent Detection
===========================================================

Uses LLM to classify user questions into intent types without
heuristic keyword matching. Supports two primary intents:

1. debug: Debugging/troubleshooting (finding problems)
2. analysis: Activity detection/analysis (finding events)
"""

from typing import Literal, Optional
from ...logger.logger import get_logger
from ...llm import Chatbot

logger = get_logger(__name__)

IntentType = Literal["debug", "analysis"]


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for AOSP log analysis queries.

Your task: Classify the user's question into ONE of these two intents:

1. **debug** - The user wants to find ERROR/PROBLEM/FAILURE locations for debugging
   - Focus: What went wrong, where is the bug/crash/error
   - Examples:
     - "The app crashed, where should I debug?"
     - "데이터 전송 실패가 발생한 로그는?"
     - "Which log range shows the memory leak?"

2. **analysis** - The user wants to LOCATE/TRACK specific activities or events
   - Focus: Where did a specific action/event occur (not errors)
   - Examples:
     - "Where did the music playback activity happen?"
     - "음악 재생이 시작된 로그는 어디야?"
     - "Locate the logs where file download occurred"

CRITICAL RULES:
- NO keyword matching, NO heuristics
- Understand the USER'S INTENT from context
- If intent is unclear, default to "analysis"
- Output ONLY: "debug" or "analysis" (nothing else)

User Question: {question}

Intent Classification:"""


async def classify_intent_with_llm(
    user_question: str,
    chatbot: Optional[Chatbot] = None
) -> IntentType:
    """Classify user question intent using LLM

    Args:
        user_question: User's input question
        chatbot: Optional LLM chatbot instance (creates new if None)

    Returns:
        IntentType: Either "debug" or "analysis"
    """
    if not user_question:
        logger.warning("[intent_classifier] Empty question, defaulting to 'analysis'")
        return "analysis"

    # Create lightweight chatbot for classification if not provided
    if chatbot is None:
        from ...utils import create_chatbot_from_env
        chatbot = create_chatbot_from_env(
            temperature=0.0,  # Deterministic classification
            max_tokens=50     # Only need 1 word output
        )

    # Build classification prompt
    prompt = INTENT_CLASSIFICATION_PROMPT.format(question=user_question)

    # 🔄 Retry up to 3 times for LLM call + response parsing
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"[intent_classifier] Attempt {attempt}/{max_attempts}: Classifying intent")

            # Get LLM classification
            response = await chatbot.ask(prompt)
            response_lower = response.strip().lower()

            # Parse response
            if "debug" in response_lower:
                intent = "debug"
            elif "analysis" in response_lower:
                intent = "analysis"
            else:
                # Default to analysis if unclear
                logger.warning(
                    f"[intent_classifier] Attempt {attempt}/{max_attempts}: Unclear LLM response: '{response}', "
                    f"defaulting to 'analysis'"
                )
                intent = "analysis"

            logger.info(
                f"[intent_classifier] Attempt {attempt}/{max_attempts}: Classification successful ✓ | "
                f"intent='{intent}' | question='{user_question[:100]}...'"
            )

            return intent

        except Exception as e:
            last_error = str(e)
            logger.error(f"[intent_classifier] Attempt {attempt}/{max_attempts}: Failed - {e}")
            if attempt < max_attempts:
                logger.info(f"[intent_classifier] Retrying classification...")
                continue

    # All attempts failed
    logger.error(
        f"[intent_classifier] ✗ Failed to classify intent after {max_attempts} attempts: {last_error}, "
        f"defaulting to 'analysis'"
    )
    return "analysis"


def get_intent_from_metadata(metadata: dict) -> Optional[IntentType]:
    """Extract explicitly specified intent from metadata

    This allows users or system to explicitly set intent without LLM classification.

    Args:
        metadata: State metadata dictionary

    Returns:
        Optional[IntentType]: Explicit intent if specified, None otherwise
    """
    if not metadata:
        return None

    # Check various metadata locations for explicit intent
    explicit_intent = (
        metadata.get("intent") or
        metadata.get("query_intent") or
        metadata.get("user_intent")
    )

    if explicit_intent and explicit_intent in ["debug", "analysis"]:
        logger.info(f"[intent_classifier] Using explicit intent from metadata: '{explicit_intent}'")
        return explicit_intent

    return None


async def determine_intent(
    user_question: str,
    metadata: dict = None,
    chatbot: Optional[Chatbot] = None,
    force_llm: bool = False
) -> IntentType:
    """Determine query intent using metadata or LLM

    Priority:
    1. Explicit intent from metadata (if not force_llm)
    2. LLM-based classification

    Args:
        user_question: User's input question
        metadata: Optional state metadata
        chatbot: Optional LLM chatbot instance
        force_llm: Force LLM classification even if metadata has intent

    Returns:
        IntentType: Determined intent ("debug" or "analysis")
    """
    # Check for explicit intent in metadata first (unless forced to use LLM)
    if not force_llm and metadata:
        explicit_intent = get_intent_from_metadata(metadata)
        if explicit_intent:
            return explicit_intent

    # Fall back to LLM classification
    return await classify_intent_with_llm(user_question, chatbot)


def get_template_name_for_intent(
    intent: IntentType,
    template_category: Literal["planner", "answer"]
) -> str:
    """Get appropriate template name based on intent and category

    Args:
        intent: The classified intent type
        template_category: Template category ("planner" or "answer")

    Returns:
        str: Template filename

    Examples:
        >>> get_template_name_for_intent("analysis", "planner")
        'planner_activity.j2'
        >>> get_template_name_for_intent("debug", "answer")
        'answer_sub.j2'
    """
    if template_category == "planner":
        if intent == "analysis":
            return "planner_activity.j2"
        else:  # debug
            return "planner_tot.j2"

    elif template_category == "answer":
        if intent == "analysis":
            return "answer_activity.j2"
        else:  # debug
            return "answer_sub.j2"

    else:
        raise ValueError(f"Unknown template category: {template_category}")
