"""
Simplified LLM Chatbot (GPT, Llama, Gemini)
============================================

LangGraph state와 함께 사용하도록 설계되었습니다.

주요 특징:
- 단순한 LLM API 인터페이스
- 세 가지 프로바이더 지원 (GPT, Llama, Gemini)
- 토큰 계산 및 분석
- 스트리밍 지원
"""

import os
import asyncio
from enum import Enum
from typing import Optional, AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
import jinja2

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import torch

# Lazy imports for optional providers (imported only when needed)
from .token_utils import TokenCounter, analyze_text_tokens
from ..logger.logger import get_logger

# 환경 변수 로드
load_dotenv()
logger = get_logger(__name__)

class LLMProvider(Enum):
    """LLM 제공업체 열거형"""
    GPT    = "gpt"      # OpenAI API
    LLAMA  = "llama"    # Together API (e.g., "meta-llama/Llama-3.1-70B-Instruct")
    GEMINI = "gemini"   # Google Gemini API (e.g., "gemini-2.5-flash")


class Chatbot:
    """단순화된 LLM 챗봇 - 순수 LLM 인터페이스"""

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.GEMINI,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.4,
        max_tokens: int = 8192,
        api_key: Optional[str] = None,
        top_p: Optional[float] = None,
    ):
        """
        Args:
            provider: LLM 제공업체 (gpt|llama|gemini)
            model: 사용할 모델명 (기본: gemini-2.0-flash-exp)
            temperature: 응답의 창의성 (0.0-1.0)
            max_tokens: 최대 토큰 수
            api_key: API 키 (없으면 환경변수에서 가져옴)
            top_p: 샘플링 top_p (옵션)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p if top_p is not None else 0.9

        # API 키 설정
        self.api_key = self._get_api_key(api_key)

        # LLM 인스턴스 생성
        self._llm = self._create_llm_instance()

        # 토큰 카운터
        self.token_counter = TokenCounter()

        logger.info(f"Chatbot initialized: {provider.value}/{model}")

    def _get_api_key(self, api_key: Optional[str]) -> str:
        """API 키 가져오기"""
        if api_key:
            return api_key

        env_keys = {
            LLMProvider.GPT:    "OPENAI_API_KEY",
            LLMProvider.LLAMA:  "TOGETHER_API_KEY",
            LLMProvider.GEMINI: "GOOGLE_API_KEY",
        }
        env_key = env_keys.get(self.provider)
        if not env_key:
            raise ValueError(f"지원하지 않는 프로바이더: {self.provider}")

        key = os.getenv(env_key)
        if not key:
            raise ValueError(f"{env_key} 환경변수가 설정되지 않았습니다.")

        return key

    def _create_llm_instance(self):
        """LLM 인스턴스 생성"""
        if self.provider == LLMProvider.LLAMA:
            # Lazy import - only load when needed
            try:
                from langchain_together import ChatTogether
            except ImportError:
                raise ImportError(
                    "langchain-together is required for Llama provider. "
                    "Install with: pip install langchain-together"
                )

            # Together API 경유
            kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "together_api_key": self.api_key,
            }
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            return ChatTogether(**kwargs)

        elif self.provider == LLMProvider.GPT:
            # Lazy import - only load when needed
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for GPT provider. "
                    "Install with: pip install langchain-openai"
                )

            # OpenAI 공식
            return ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        elif self.provider == LLMProvider.GEMINI:
            # Lazy import - only load when needed
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "langchain-google-genai is required for Gemini provider. "
                    "Install with: pip install langchain-google-genai"
                )

            # Google Gemini API
            # Note: Gemini uses 'max_output_tokens' not 'max_tokens'
            # Limit thinking/reasoning tokens to preserve quality while ensuring content output
            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,  # Gemini-specific parameter (8192)
                thinking_budget=1024,  # Limit reasoning to 1024 tokens (allows reasoning but saves 7168 for content)
            )

        else:
            raise ValueError(f"지원하지 않는 프로바이더: {self.provider}")

    async def ask(
        self,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        단일 질문에 대한 응답 텍스트 반환
        """
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(
                HumanMessage(
                    content=question,
                    kwargs={
                        "agent_name": "human",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )
            response = await self._llm.ainvoke(messages)

            # Debug: Log full response for empty content
            if not response.content:
                logger.error(f"Empty response from LLM!")
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response attributes: {dir(response)}")
                if hasattr(response, 'response_metadata'):
                    logger.error(f"Response metadata: {response.response_metadata}")
                if hasattr(response, 'usage_metadata'):
                    logger.error(f"Usage metadata: {response.usage_metadata}")
                logger.error(f"Full response object: {response}")

            logger.debug(f"LLM response generated: {len(response.content)} characters")
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def invoke_messages(
        self,
        messages: list[BaseMessage],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        메시지 리스트(컨텍스트)로 LLM 호출
        """
        try:
            if system_prompt:
                response = await self._llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
            else:
                response = await self._llm.ainvoke(messages)
            logger.debug("Message-based LLM call ends")
            return response.content
        except Exception as e:
            logger.error(f"Message-based LLM call failed: {e}")
            raise

    async def ask_stream(
        self,
        question: str,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 응답 생성
        """
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(
                HumanMessage(
                    content=question,
                    kwargs={
                        "agent_name": "human",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Streaming call failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return self.token_counter.count_tokens(text, self.model)

    def analyze_tokens(self, text: str) -> dict:
        """텍스트의 토큰 분석"""
        return analyze_text_tokens(text, self.model)

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
