"""
CaucowChat Graph Module - LangGraph Integration
=============================================

LangGraph StateGraph를 직접 활용하는 그래프 구성 유틸리티를 제공합니다.

"""

from typing import Callable
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from .state import AgentState, get_last_message, update_state_metadata
from ..llm import Chatbot

from datetime import datetime, timezone


def create_agent_graph() -> StateGraph:
    """기본 에이전트 그래프 생성"""
    return StateGraph(AgentState)


def llm_node(
    system_prompt: str,
    chatbot: Chatbot = None,
    agent_name: str = "assistant",
    context: bool = True
) -> Callable[[AgentState], AgentState]:
    """LLM 노드 함수 생성기

    Args:
        system_prompt: 시스템 프롬프트
        chatbot: LLM 챗봇 인스턴스 (None이면 Friendli 기본 생성)
        agent_name: 에이전트 이름
        context: 컨텍스트(대화 맥락) 유지 여부

    Returns:
        LangGraph 노드 함수
    """
    if chatbot is None:
        from ..utils import create_chatbot_from_env
        chatbot = create_chatbot_from_env(
            temperature=0.4
            # max_tokens는 기본값(4096) 사용
        )
    
    async def node_function(state: AgentState) -> AgentState:
        """실제 노드 실행 함수"""
        # Provider 정보를 metadata에 저장 (retrieve node에서 사용)
        metadata = state.get("metadata", {})
        if "llm_provider" not in metadata:
            metadata["llm_provider"] = chatbot.provider.value if chatbot else "unknown"

        # 마지막 메시지 가져오기
        last_message = get_last_message(state)
        if not last_message:
            return state

        try:
            # LLM 호출
            if context:
                # 컨텍스트 유지
                response = await chatbot.invoke_messages(
                    messages=state["messages"],
                    system_prompt=system_prompt
                )
            else:
                # 컨텍스트 미유지
                response = await chatbot.ask(
                    question=last_message.content,
                    system_prompt=system_prompt
                )

            # 응답 메시지 생성
            ai_message = AIMessage(
                content=response,
                kwargs={
                    "agent_name": agent_name,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # 상태 업데이트 (add_messages가 자동으로 메시지 추가)
            return AgentState(
                messages=[ai_message],  # add_messages가 기존 메시지에 추가
                metadata=update_state_metadata(
                    state,
                    last_agent=agent_name,
                    processed=True
                ),
                current_agent=agent_name,
                session_id=state.get("session_id", "")
            )
            
        except Exception as e:
            # 에러 처리
            error_message = AIMessage(
                content=f"Error occurred: {str(e)}",
                kwargs={
                    "agent_name": agent_name,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            return AgentState(
                messages=[error_message],
                metadata=update_state_metadata(
                    state,
                    error=str(e),
                    last_agent=agent_name
                ),
                current_agent=agent_name,
                session_id=state.get("session_id", "")
            )
    
    return node_function
