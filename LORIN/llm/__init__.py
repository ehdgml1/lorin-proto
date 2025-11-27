"""
CaucowChat LLM Module - Multi-Provider Language Model Interface
===============================================================

다양한 LLM 제공업체(Gemini, Claude, OpenAI)와의 통합 인터페이스를 제공하는 모듈입니다.
LangGraph 기반 Agent 모듈과의 원활한 통합을 지원하며, 토큰 관리 기능을 제공합니다.

모듈 구조
--------
```
llm/
├── chatbot.py       # 단순화된 LLM 챗봇 인터페이스
├── token_utils.py   # 토큰 계산 및 분석 유틸리티
└── __init__.py     # 공개 API 정의
```

주요 컴포넌트
-----------
- **Chatbot**: 단순화된 LLM 챗봇 인터페이스
- **LLMProvider**: 지원하는 LLM 제공업체 열거형
- **TokenCounter**: 모델별 토큰 계산 및 분석
- **ModelLimits**: 모델별 토큰 제한 정보
- **analyze_text_tokens**: 텍스트 토큰 분석 함수

지원 모델
--------
- **Gemini**: 2.5-flash, 2.5-pro, 1.5-pro, 1.5-flash
- **Claude**: 3.5-sonnet, 3-opus, 3-haiku
- **OpenAI**: gpt-4, gpt-4-turbo, gpt-3.5-turbo

"""

# Core classes
from .chatbot import Chatbot, LLMProvider
from .token_utils import TokenCounter, ModelLimits, get_gemini_limits, analyze_text_tokens

__all__ = [
    # Core classes
    "Chatbot",
    "LLMProvider",
    
    # Token utilities
    "TokenCounter",
    "ModelLimits",
    "get_gemini_limits",
    "analyze_text_tokens",
]

__version__ = "0.1.0"
