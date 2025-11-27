"""
Chatbot Factory Module
======================
환경변수 기반 Chatbot 인스턴스 생성 유틸리티
"""

import os
from ..llm import Chatbot, LLMProvider
from ..logger.logger import get_logger

logger = get_logger(__name__)


def create_chatbot_from_env(
    temperature: float = 0.4,
    max_tokens: int = 1024,
    override_provider: LLMProvider = None,
    override_model: str = None,
) -> Chatbot:
    """
    환경변수 기반 Chatbot 인스턴스 생성

    Args:
        temperature: LLM temperature (기본값: 0.4)
        max_tokens: 최대 토큰 수 (기본값: 1024)
        override_provider: 프로바이더 오버라이드 (None이면 환경변수 사용)
        override_model: 모델 오버라이드 (None이면 환경변수 사용)

    Returns:
        Chatbot 인스턴스

    Raises:
        ValueError: 잘못된 프로바이더 설정 시
    """
    # 환경변수에서 provider와 model 읽기
    provider_str = os.getenv("LLM_PROVIDER", "exaone").upper()
    model_name = os.getenv("LLM_MODEL", "exaone-4.0.1-32b")

    # 오버라이드가 있으면 사용
    if override_provider is not None:
        provider = override_provider
    else:
        try:
            provider = LLMProvider[provider_str]
        except KeyError:
            logger.error(
                f"Invalid LLM_PROVIDER: {provider_str}. "
                f"Valid options: {', '.join([p.name for p in LLMProvider])}. "
                f"Falling back to EXAONE."
            )
            provider = LLMProvider.EXAONE
            model_name = "exaone-4.0.1-32b"

    if override_model is not None:
        model_name = override_model

    # Chatbot 인스턴스 생성
    chatbot = Chatbot(
        provider=provider,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    logger.debug(
        f"Chatbot created from env: {provider.value}/{model_name} "
        f"(temp={temperature}, max_tokens={max_tokens})"
    )

    return chatbot
