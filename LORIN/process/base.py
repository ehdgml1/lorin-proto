"""
Process Base Module - Data Processing and Workflow Management
============================================================

Agent와 LLM을 통합하여 복잡한 데이터 처리 워크플로우를 관리하는 모듈입니다.
다양한 처리 패턴을 지원하며 확장 가능한 아키텍처를 제공합니다.

주요 클래스
----------
- None

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
    25.08.17. 이성훈

    - 싱글턴 대화 메서드입니다.
    - 질문별로 새로운 상태를 생성하여 그래프를 실행합니다.
    - 최종 답변은 messages 리스트에 저장됩니다.
    """
    logger = get_logger(__name__)
    logger.debug("Single-turn conversation begins")

    states = []

    for i, question in enumerate(questions, 1):
        # 초기 상태 생성 (실험 config 포함)
        metadata = {
            "question_id": i,
            "experiment_config": config or {}  # 실험 설정 전달
        }
        current_state = create_initial_state(
            question,
            metadata=metadata
        )

        logger.info(f"[Q{i}] {question}")
        
        try:
            # 그래프 실행
            current_state = await app.ainvoke(current_state)
            
            # 결과 출력
            last_message = get_last_message(current_state)
            logger.info(f"[A{i}] {last_message.content}")

            states.append(current_state)
   
        except Exception as e:
            logger.error(f"Question {i} processing failed: {e}")

    logger.debug("Single-turn conversation ends")

    return states


async def _multi_turn_conversation(app: StateGraph, questions: list[str], config: Optional[dict] = None) -> str:
    """
    25.08.17. 이성훈

    - 멀티턴 대화 메서드입니다.
    - 모든 질문에 대해 하나의 상태를 공유하여 그래프를 실행합니다.
    - 최종 답변은 messages 리스트에 저장됩니다.
    """
    logger = get_logger(__name__)
    logger.debug("Multi-turn conversation begins")

    # 초기 상태 생성 (실험 config 포함)
    metadata = {
        "experiment_config": config or {}  # 실험 설정 전달
    }
    current_state = create_initial_state(
        questions[0],
        metadata=metadata
    )

    for i, question in enumerate(questions, 1):
        logger.info(f"[Q{i}] {question}")

        # 첫 번째가 아니면 새 메시지 추가
        if i > 1:
            # 이전 상태에 새 사용자 메시지 추가
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
            # 그래프 실행
            current_state = await app.ainvoke(current_state)
            
            # 결과 출력
            last_message = get_last_message(current_state)
            logger.info(f"[A{i}] {last_message.content}")

        except Exception as e:
            logger.error(f"Question {i} processing failed: {e}")

    logger.debug("Multi-turn conversation ends")

    return current_state
    

async def _llm_process(app: StateGraph, config: Optional[dict] = None, question: Optional[str] = None):
    """
    25.08.17. 이성훈

    - 현재 솔루션 버전에서는 사용자 상호작용을 고려하고 있지 않습니다.
    - LLM 사용을 위해서는 questions 리스트에 질문을 추가하면 됩니다.
    - 향후 파일을 읽어오거나 API를 통해 사용자 상호작용을 하도록 수정될 예정입니다.
    - 답변 생성 과정에서 싱글턴과 멀티턴 대화를 선택할 수 있습니다.
    """
    logger = get_logger(__name__)

    logger.debug("LLM process begins")

    # 1. 질문 준비
    # ✨ 두 가지 의도(intent) 지원:
    # - debug: 문제/에러 디버깅을 위한 로그 범위 찾기
    # - analysis: 특정 활동/이벤트가 발생한 로그 위치 찾기

    if question:
        # 🔬 실험 모드: 외부에서 질문 제공 (experiment_runner.py에서 사용)
        questions = [question]
        logger.info(f"Using provided question: {question[:100]}...")
    else:
        # 기본 모드: 하드코딩된 질문 사용 (테스트/개발용)
        questions = [
            "The data passed between processes is too large and this is causing a failure. Please tell me which log range I should check for debugging."
            #"When does the Android system boot process start the DropBoxManager service?"
        ]
        logger.info("Using default hardcoded question (development mode)")
    
    # 2. 답변 생성
    conversation_type = "single"

    if conversation_type == "single":
        states = await _single_turn_conversation(app, questions, config)
    elif conversation_type == "multi":
        state = await _multi_turn_conversation(app, questions, config)
        states = [state]
    else:
        logger.warning("Invalid conversation type")
        states = await _single_turn_conversation(app, questions, config)

    # 3. 답변 처리
    # 25.08.17. 이성훈: 디버깅을 위한 코드입니다.
    # 디버깅을 위해 모든 state를 출력하고 싶다면 주석을 해제하세요.
    # 향후 다른 코드로 대체될 예정입니다.
    for state in states:
        logger.info(f'{format_agent_state(state)}')

    logger.debug("LLM process ends")
    
    return states


async def main_process(
    chatbot: Chatbot,
    vectorstore,
    *,
    sparse_store=None,                 # ← 추가
    corpus_path: Optional[str] = None,  # ← 추가
    config: Optional[dict] = None,      # ← 실험 설정
    question: Optional[str] = None      # ← 실험용 질문
):
    logger = get_logger(__name__)
    logger.debug("Main process begins")

    if config:
        logger.info(f"Main process running with config: {config}")
    if question:
        logger.info(f"Main process running with question: {question[:100]}...")

    graph = create_agent_graph()
    logger.debug("1.Gragh created")

    # 기존: graph = initialize_graph(graph, chatbot, vectorstore)
    graph = await initialize_graph(
        graph,
        chatbot,
        vectorstore,
        corpus_path=corpus_path,   # ← 추가
        sparse_store=sparse_store,  # ← 추가
        config=config               # ← 실험 설정 전달
    )
    logger.debug("2.Graph initialized")

    app = graph.compile()
    logger.debug("3.Graph compiled")

    states = await _llm_process(app, config, question)  # ← question 전달
    logger.debug("4.LLM process ends")
    return states