"""
Question Decomposer Module - Parse Question Structure for Log Analysis
======================================================================

Uses LLM reasoning to identify primary vs secondary triggers in questions,
avoiding all heuristics and hardcoded keywords.

Key Principles:
- NO pattern matching or if-then rules
- NO domain-specific keywords (e.g., no "StorageManagerService")
- Uses linguistic analysis (specificity assessment)
- Generalizable to any "A or B" question structure
"""

from typing import Dict, Optional
from ...logger.logger import get_logger
from ...llm import Chatbot

logger = get_logger(__name__)


DECOMPOSITION_PROMPT = """You are a question structure analyzer for log analysis queries.

Your task: Parse the question structure using linguistic reasoning to identify primary vs secondary triggers.

# Core Principle: Specificity Assessment

In questions with multiple events/conditions (especially "A or B" structure):
1. Identify all mentioned events/conditions
2. Assess specificity: Which terms are more specific vs more general?
3. More specific terms are usually PRIMARY (they define the context)
4. More general terms are usually SECONDARY (they describe alternatives)

# Specificity Analysis Framework

**More Specific** (usually primary):
- Targets a particular operation or state (e.g., "user switching", "notification received")
- Has narrower semantic scope
- Occurs in a specific context or phase

**More General** (usually secondary):
- Applies to broader system behavior (e.g., "restarting", "boot")
- Has wider semantic scope
- Can occur in multiple contexts

# Context Inference

Based on the PRIMARY trigger, infer the operational context:
- **user_session**: User account operations (login, logout, unlock, switch)
- **device_boot**: System initialization and startup sequences
- **app_runtime**: Application execution and lifecycle
- **system_service**: Background system service operations
- **unknown**: Cannot determine from question alone

# Temporal Inference

Based on problem characteristics, estimate when evidence appears:
- **early**: Beginning of log (0-30%) - typically boot/initialization issues
- **mid**: Middle of log (30-70%) - typically runtime/session issues
- **late**: End of log (70-100%) - typically shutdown/resource exhaustion issues
- **unknown**: Evidence could appear anywhere

# Output Format

Return ONLY a JSON object. Do NOT include any explanatory text.

Start immediately with the opening brace and end with closing brace.

Format:
{{
  "primary_issue": "core problem",
  "primary_trigger": "most specific trigger",
  "secondary_triggers": ["other triggers"],
  "context_scope": "user_session|device_boot|app_runtime|system_service|unknown",
  "temporal_hint": "early|mid|late|unknown",
  "reasoning": "your analysis"
}}

# Examples

**Example 1**: "App crashes after receiving notification OR during reboot"

Analysis:
1. Events identified: "receiving notification" (specific app event), "during reboot" (general system event)
2. Specificity: "receiving notification" targets specific app runtime context vs "reboot" applies broadly
3. Primary: "receiving notification" (more specific)
4. Context: app_runtime (notification handling occurs during app execution)
5. Temporal: mid-to-late (app runtime issues appear after initialization)

Output:
{{
  "primary_issue": "app crash",
  "primary_trigger": "receiving notification",
  "secondary_triggers": ["during reboot"],
  "context_scope": "app_runtime",
  "temporal_hint": "mid",
  "reasoning": "Notification is more specific (targets app runtime) than reboot (general system event). Primary trigger defines context as app_runtime. App crashes typically manifest during runtime operation."
}}

**Example 2**: "System service restart after user switching OR system startup"

Analysis:
1. Events: "user switching" (specific user operation), "system startup" (general boot process)
2. Specificity: "user switching" targets specific user session context vs "startup" is broader
3. Primary: "user switching" (more specific)
4. Context: user_session (user switching is a session management operation)
5. Temporal: mid (user session operations occur during runtime, not at boot)

Output:
{{
  "primary_issue": "system service restart",
  "primary_trigger": "user switching",
  "secondary_triggers": ["system startup"],
  "context_scope": "user_session",
  "temporal_hint": "mid",
  "reasoning": "User switching is more specific (targets user session operations) than system startup (general boot). Primary trigger indicates user_session context. Session-related issues typically appear mid-log during runtime."
}}

**Example 3**: "Memory leak causing slowdown"

Analysis:
1. Single trigger: "memory leak"
2. No "A or B" structure
3. Primary: "memory leak"
4. Context: app_runtime (memory leaks occur during execution)
5. Temporal: late (leaks accumulate over time)

Output:
{{
  "primary_issue": "slowdown due to memory leak",
  "primary_trigger": "memory leak",
  "secondary_triggers": [],
  "context_scope": "app_runtime",
  "temporal_hint": "late",
  "reasoning": "Single trigger scenario. Memory leaks are runtime execution issues that manifest after extended operation. Context is app_runtime, temporal is late as leaks accumulate."
}}

# Critical Instructions

1. Use REASONING, not pattern matching
2. Assess specificity by analyzing semantic scope and operational context
3. Context is INFERRED from primary trigger's domain, not hardcoded
4. NO specific component names (e.g., avoid "StorageManagerService", use "storage management")
5. Focus on what makes one trigger more specific than another
6. Output pure JSON only with opening and closing braces

---

User Question: {question}

Analyze and output JSON:"""


async def decompose_question(
    user_question: str,
    chatbot: Optional[Chatbot] = None
) -> Dict:
    """Decompose question to identify primary vs secondary triggers using LLM reasoning

    This function uses linguistic analysis (not heuristics) to parse question structure.
    It identifies which triggers are more specific (primary) vs more general (secondary).

    Args:
        user_question: User's input question
        chatbot: Optional LLM chatbot instance (creates new if None)

    Returns:
        Dict with structure:
        {
            "primary_issue": str,
            "primary_trigger": str,
            "secondary_triggers": List[str],
            "context_scope": str,
            "temporal_hint": str,
            "reasoning": str
        }
    """
    if not user_question:
        logger.warning("[question_decomposer] Empty question, returning safe defaults")
        return {
            "primary_issue": "",
            "primary_trigger": "",
            "secondary_triggers": [],
            "context_scope": "unknown",
            "temporal_hint": "unknown",
            "reasoning": "Empty question provided"
        }

    # Create lightweight chatbot for decomposition if not provided
    if chatbot is None:
        logger.info("[question_decomposer] Creating new chatbot instance for decomposition")
        from ...utils import create_chatbot_from_env
        try:
            chatbot = create_chatbot_from_env(
                temperature=0.0,  # Deterministic analysis
                max_tokens=1000    # Sufficient for analysis + JSON
            )
            logger.info(f"[question_decomposer] Chatbot created: type={type(chatbot).__name__}")
        except Exception as e:
            logger.error(f"[question_decomposer] Failed to create chatbot: {e}")
            # Return safe defaults immediately
            return {
                "primary_issue": user_question,
                "primary_trigger": "unknown",
                "secondary_triggers": [],
                "context_scope": "unknown",
                "temporal_hint": "unknown",
                "reasoning": f"Failed to create chatbot: {e}"
            }

    # Build decomposition prompt
    prompt = DECOMPOSITION_PROMPT.format(question=user_question)

    logger.info(f"[question_decomposer] Built prompt ({len(prompt)} chars)")
    logger.debug(f"[question_decomposer] Full prompt:\n{prompt}")

    # Retry up to 3 times for robust LLM call + JSON parsing
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"[question_decomposer] === Attempt {attempt}/{max_attempts} ===")

            # Get LLM analysis
            response = await chatbot.ask(prompt)
            logger.info(f"[question_decomposer] Response type: {type(response)}")
            logger.info(f"[question_decomposer] Response object: {response}")

            # Handle different response types
            if hasattr(response, 'content'):
                raw = str(response.content).strip()
            else:
                raw = str(response).strip()

            logger.info(f"[question_decomposer] Raw response length: {len(raw)} chars")
            logger.info(f"[question_decomposer] Raw response (first 300 chars): {raw[:300]}")
            logger.info(f"[question_decomposer] Raw response (last 200 chars): {raw[-200:]}")

            # Preprocess: Fix missing opening brace (common LLM issue)
            raw_stripped = raw.strip()
            if not raw_stripped.startswith('{') and '"primary_issue"' in raw_stripped:
                logger.warning("[question_decomposer] Response missing opening brace, attempting repair")
                # Find the first quote or field name and add opening brace
                raw_stripped = '{' + raw_stripped
                logger.info(f"[question_decomposer] Repaired response preview: {raw_stripped[:200]}")

            # Extract JSON from response
            from ...utils.json_parser import extract_json_from_response
            decomposition = extract_json_from_response(raw_stripped)

            if decomposition:
                # Validate required fields
                required_fields = ["primary_issue", "primary_trigger", "context_scope", "reasoning"]
                if all(field in decomposition for field in required_fields):
                    logger.info(f"[question_decomposer] ✓ Decomposition successful")
                    logger.info(f"  Primary Issue: {decomposition.get('primary_issue', 'N/A')}")
                    logger.info(f"  Primary Trigger: {decomposition.get('primary_trigger', 'N/A')}")
                    logger.info(f"  Context Scope: {decomposition.get('context_scope', 'N/A')}")
                    logger.info(f"  Temporal Hint: {decomposition.get('temporal_hint', 'N/A')}")

                    # Ensure default values for optional fields
                    decomposition.setdefault("secondary_triggers", [])
                    decomposition.setdefault("temporal_hint", "unknown")

                    return decomposition
                else:
                    logger.warning(
                        f"[question_decomposer] Attempt {attempt}: Missing required fields. "
                        f"Got: {list(decomposition.keys())}"
                    )
            else:
                logger.warning(f"[question_decomposer] Attempt {attempt}: JSON extraction failed")
                logger.warning(f"[question_decomposer] Raw response (first 500 chars): {raw[:500]}")
                logger.warning(f"[question_decomposer] Raw response (last 200 chars): {raw[-200:]}")

            if attempt < max_attempts:
                logger.info(f"[question_decomposer] Retrying with fresh LLM call...")
                continue

        except Exception as e:
            last_error = str(e)
            logger.error(f"[question_decomposer] Attempt {attempt}/{max_attempts}: Exception - {e}")
            logger.error(f"[question_decomposer] Exception type: {type(e).__name__}")
            logger.error(f"[question_decomposer] Raw response causing error: {raw[:500] if 'raw' in locals() else 'N/A'}")
            if attempt < max_attempts:
                logger.info(f"[question_decomposer] Retrying after exception...")
                continue

    # All attempts failed - return safe defaults
    logger.error(
        f"[question_decomposer] ✗ Failed to decompose question after {max_attempts} attempts: {last_error}"
    )
    logger.warning("[question_decomposer] Returning safe defaults (no decomposition)")

    return {
        "primary_issue": user_question,  # Use full question as fallback
        "primary_trigger": "unknown",
        "secondary_triggers": [],
        "context_scope": "unknown",
        "temporal_hint": "unknown",
        "reasoning": f"Failed to decompose after {max_attempts} attempts. Using question as-is."
    }
