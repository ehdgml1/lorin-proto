"""
CaucowChat State Module - LangGraph State Management
==================================================

LangGraph의 TypedDict 기반 상태 관리를 제공하는 모듈입니다.

주요 특징:
- TypedDict 기반 상태 정의
- add_messages를 통한 메시지 상태 관리
- 메타데이터 및 사용자 정의 필드 지원
- LangGraph StateGraph와 완벽 통합

"""

from typing import TypedDict, Annotated, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime, timezone

def replace_reducer(_, new):
    # LangGraph 리듀서: 새 값으로 교체
    return new

class AgentState(TypedDict):
    """LangGraph StateGraph에서 사용할 에이전트 상태

    LangGraph의 TypedDict 패턴을 따르는 상태 정의입니다.
    add_messages를 사용하여 메시지 상태를 자동으로 관리합니다.

    Fields:
        messages: 대화 메시지 리스트 (자동 병합)
        metadata: 사용자 정의 메타데이터
        current_agent: 현재 처리 중인 에이전트 이름
        session_id: 세션 식별자 (선택적)
        experiment_output: 실험용 구조화된 출력 (선택적)
    """
    # LangGraph의 add_messages를 사용하여 메시지 자동 관리
    messages: Annotated[list[BaseMessage], add_messages]          # 히스토리(append)
    context_messages: Annotated[list[BaseMessage], replace_reducer]  # 모델 입력용(교체)
    metadata: dict[str, Any]
    current_agent: str
    session_id: str
    experiment_output: Optional[dict[str, Any]]  # 실험용 구조화된 출력

def format_agent_state(state: AgentState) -> str:
    """AgentState를 읽기 좋은 형태로 포맷팅"""
    lines = ["[AgentState]"]
    
    # 기본 정보
    lines.append(f"Current Agent: {state.get('current_agent', 'None')}")
    lines.append(f"Session ID: {state.get('session_id', 'None')}")
    lines.append(f"Messages: {len(state.get('messages', []))}")
    lines.append("")
    
    # 메시지 히스토리
    for i, msg in enumerate(state.get("messages", []), 1):
        ak = _ak(msg)  # ← additional_kwargs 또는 kwargs 중 존재하는 쪽을 안전하게 읽음
        agent_name = ak.get('agent_name', 'unknown')
        timestamp = ak.get('timestamp', 'unknown')

        # pretty 출력이 실패하면 content로 폴백
        try:
            pretty = msg.pretty_repr()
        except Exception:
            pretty = str(getattr(msg, "content", "") or "")

        lines.append(
            f"[Message #{i:0>3d} from {agent_name:<20}: {timestamp}]\n{pretty}\n"
        )
    
    # 메타데이터
    if state.get("metadata"):
        lines.append("")
        lines.append("Metadata:")
        for key, value in state["metadata"].items():
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


# 편의를 위한 상태 생성 함수들
def create_initial_state(
    initial_message: str | BaseMessage,
    metadata: dict[str, Any] = None,
    session_id: str = None
) -> AgentState:
    """초기 상태 생성 헬퍼 함수"""
    from langchain_core.messages import HumanMessage
    
    if isinstance(initial_message, str):
        message = HumanMessage(
            content=initial_message,
            additional_kwargs={           # ← 통일: additional_kwargs 사용
                "agent_name": "human",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    else:
        message = initial_message

    # 🔬 실험용 메트릭 초기화
    init_metadata = metadata or {}
    if "faiss_calls" not in init_metadata:
        init_metadata["faiss_calls"] = 0
    if "planner_iterations" not in init_metadata:
        init_metadata["planner_iterations"] = 0

    # 🔄 Sequential evidence expansion을 기본값으로 설정
    # (명시적으로 다른 phase를 설정하지 않는 한 "initial"로 시작)
    if "planner" not in init_metadata:
        init_metadata["planner"] = {}
    if "phase" not in init_metadata["planner"]:
        init_metadata["planner"]["phase"] = "initial"  # Sequential mode 기본값

    return AgentState(
        messages=[message],
        context_messages=[message],       # ← 새로 추가
        metadata=init_metadata,
        current_agent="",
        session_id=session_id or "",
        experiment_output=None  # 실험용 출력 (초기값 None)
    )


def update_state_metadata(state: AgentState, **kwargs) -> dict[str, Any]:
    """상태 메타데이터 업데이트 헬퍼"""
    new_metadata = state["metadata"].copy()
    new_metadata.update(kwargs)
    return new_metadata


# 상태 검증 함수들
def has_messages(state: AgentState) -> bool:
    """메시지가 있는지 확인"""
    return len(state.get("messages", [])) > 0


def get_last_message(state: AgentState) -> BaseMessage | None:
    """마지막 메시지 가져오기"""
    messages = state.get("messages", [])
    return messages[-1] if messages else None


def get_last_human_message(state: AgentState) -> BaseMessage | None:
    """마지막 사용자 메시지 가져오기"""
    from langchain_core.messages import HumanMessage
    
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None

def _ak(m) -> dict:
    return (getattr(m, "additional_kwargs", {}) or getattr(m, "kwargs", {}) or {})

def format_graph_result(result: dict) -> str:
    """LangGraph 실행 결과를 예쁘게 포맷팅"""
    lines = ["=" * 80]
    lines.append("GRAPH 실행 결과")
    lines.append("=" * 80)
    
    # 기본 정보
    metadata = result.get('metadata', {})
    lines.append(f"📊 Session: {metadata.get('session', 'Unknown')}")
    lines.append(f"🔍 Question ID: {metadata.get('question_id', 'N/A')}")
    lines.append(f"👤 Current Agent: {result.get('current_agent', 'Unknown')}")
    lines.append(f"🆔 Session ID: {result.get('session_id', 'None')}")
    lines.append(f"✅ Processed: {metadata.get('processed', False)}")
    lines.append("")
    
    # 메시지 히스토리
    messages = result.get("messages", [])
    lines.append(f"💬 CONVERSATION HISTORY ({len(messages)} messages)")
    lines.append("-" * 80)
    
    for i, msg in enumerate(messages, 1):
        # 메시지 헤더
        agent_name = msg.kwargs.get('agent_name', 'unknown')
        timestamp = msg.kwargs.get('timestamp', 'unknown')
        msg_type = msg.__class__.__name__
        
        # 에이전트별 이모지
        emoji_map = {
            'human': '👤',
            'db_agent': '🗄️',
            'summary': '📋',
            'evaluator': '🔍',
            'assistant': '🤖'
        }
        emoji = emoji_map.get(agent_name, '🔹')
        
        lines.append(f"{emoji} Message #{i:03d} | {msg_type} | {agent_name} | {timestamp}")
        lines.append("-" * 60)
        
        # 메시지 내용 처리
        content = msg.content
        if isinstance(content, list):
            # 리스트 형태의 content 처리 (여러 단계가 포함된 경우)
            for j, step_content in enumerate(content):
                if step_content.strip():
                    lines.append(f"  📝 Step {j+1}:")
                    # 긴 내용은 줄바꿈 처리
                    for line in str(step_content).split('\n'):
                        lines.append(f"    {line}")
                    lines.append("")
        else:
            # 단일 문자열 content 처리
            if content.strip():
                # JSON 형태인지 확인
                if content.strip().startswith('{') and content.strip().endswith('}'):
                    lines.append("  📊 JSON Response:")
                    for line in str(content).split('\n'):
                        lines.append(f"    {line}")
                else:
                    lines.append("  💭 Content:")
                    for line in str(content).split('\n'):
                        lines.append(f"    {line}")
        
        lines.append("")
    
    # 메타데이터 상세
    if metadata:
        lines.append("📋 METADATA DETAILS")
        lines.append("-" * 40)
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("🏁 END OF RESULT")
    lines.append("=" * 80)
    
    return "\n".join(lines)