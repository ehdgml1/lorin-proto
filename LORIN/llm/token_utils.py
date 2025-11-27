"""
토큰 유틸리티 모듈

다양한 LLM 모델의 토큰 계산 및 제한 정보를 제공합니다.

토큰화 라이브러리 현황:
- OpenAI (GPT): tiktoken 공식 라이브러리 사용 (정확)
- Google Gemini: 공식 라이브러리 없음 (GPT-4 기준 추정)
- Anthropic Claude: 공식 라이브러리 없음 (GPT-4 기준 추정)

주의: Gemini와 Claude는 실제 토큰 수와 차이가 있을 수 있습니다.
정확한 토큰 수가 필요한 경우 각 모델의 API를 직접 사용하세요.
"""

import tiktoken
from ..logger.logger import get_logger

logger = get_logger(__name__)


class ModelLimits:
    """모델별 토큰 제한 정보"""
    
    # Gemini 모델
    GEMINI_2_5_FLASH = {
        "max_input_tokens": 1_000_000,  # 1M 토큰
        "max_output_tokens": 65_000,    # 65K 토큰 (64K + 여유)
        "context_window": 1_000_000,    # 1M 토큰
        "model_name": "gemini-2.5-flash",
        "release_date": "2025-04-17",
        "knowledge_cutoff": "2025-01"
    }
    
    GEMINI_2_5_PRO = {
        "max_input_tokens": 1_000_000,  # 1M 토큰 (곧 2M으로 확장 예정)
        "max_output_tokens": 64_000,    # 64K 토큰
        "context_window": 1_000_000,    # 1M 토큰 (곧 2M으로 확장 예정)
        "model_name": "gemini-2.5-pro",
        "release_date": "2025-03-25",
        "knowledge_cutoff": "2025-01"
    }
    
    GEMINI_2_0_FLASH = {
        "max_input_tokens": 1_000_000,  # 1M 토큰
        "max_output_tokens": 8_192,     # 8K 토큰
        "context_window": 1_000_000,    # 1M 토큰
        "model_name": "gemini-2.0-flash",
        "release_date": "2024-12-11",
        "knowledge_cutoff": "2024-08"
    }
    
    GEMINI_1_5_FLASH = {
        "max_input_tokens": 1_000_000,  # 1M 토큰
        "max_output_tokens": 8_192,     # 8K 토큰
        "context_window": 1_000_000,    # 1M 토큰
        "model_name": "gemini-1.5-flash",
        "release_date": "2024-05-14",
        "knowledge_cutoff": "2024-08"
    }
    
    GEMINI_1_5_PRO = {
        "max_input_tokens": 1_000_000,  # 1M 토큰
        "max_output_tokens": 8_192,     # 8K 토큰
        "context_window": 1_000_000,    # 1M 토큰
        "model_name": "gemini-1.5-pro",
        "release_date": "2024-02-15",
        "knowledge_cutoff": "2024-08"
    }
    
    # OpenAI 모델
    GPT_4_TURBO = {
        "max_input_tokens": 128_000,
        "max_output_tokens": 4096,
        "context_window": 128_000,
        "model_name": "gpt-4-turbo"
    }
    
    GPT_4O = {
        "max_input_tokens": 128_000,
        "max_output_tokens": 16384,
        "context_window": 128_000,
        "model_name": "gpt-4o"
    }
    
    # Claude 모델
    CLAUDE_3_5_SONNET = {
        "max_input_tokens": 200_000,
        "max_output_tokens": 8192,
        "context_window": 200_000,
        "model_name": "claude-3-5-sonnet-20241022"
    }
    
    @classmethod
    def get_model_limits(cls, model_name: str) -> dict:
        """모델 이름으로 제한 정보 조회"""
        model_map = {
            # Gemini 2.5 모델
            "gemini-2.5-flash": cls.GEMINI_2_5_FLASH,
            "gemini-2.5-flash-exp": cls.GEMINI_2_5_FLASH,
            "gemini-2.5-pro": cls.GEMINI_2_5_PRO,
            "gemini-2.5-pro-exp": cls.GEMINI_2_5_PRO,
            # Gemini 2.0 모델
            "gemini-2.0-flash": cls.GEMINI_2_0_FLASH,
            "gemini-2.0-flash-exp": cls.GEMINI_2_0_FLASH,
            # Gemini 1.5 모델
            "gemini-1.5-flash": cls.GEMINI_1_5_FLASH,
            "gemini-1.5-pro": cls.GEMINI_1_5_PRO,
            "gemini-pro": cls.GEMINI_1_5_PRO,  # 기본값
            # OpenAI 모델
            "gpt-4-turbo": cls.GPT_4_TURBO,
            "gpt-4o": cls.GPT_4O,
            # Claude 모델
            "claude-3-5-sonnet": cls.CLAUDE_3_5_SONNET,
        }
        
        return model_map.get(model_name.lower(), {
            "max_input_tokens": 4096,
            "max_output_tokens": 2048,
            "context_window": 4096,
            "model_name": model_name
        })


class TokenCounter:
    """토큰 계산 유틸리티"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._encoders = {}
    
    def _get_encoder(self, model_name: str):
        """모델에 맞는 토큰 인코더 가져오기"""
        if model_name in self._encoders:
            return self._encoders[model_name]
        
        try:
            if "gpt" in model_name.lower():
                # OpenAI 모델 - 정확한 토큰화 지원
                if "gpt-4" in model_name.lower():
                    encoder = tiktoken.encoding_for_model("gpt-4")
                else:
                    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Gemini, Claude 등 - 공식 토크나이저 없음
                # GPT-4와 유사한 패턴으로 근사치 계산
                self.logger.info(f"{model_name}은 공식 토크나이저가 없어 GPT-4 기준으로 추정합니다")
                encoder = tiktoken.encoding_for_model("gpt-4")
            
            self._encoders[model_name] = encoder
            
            return encoder
            
        except Exception as e:
            self.logger.warning(f"토큰 인코더 로드 실패: {e}, 기본 인코더 사용")
            # 기본 인코더 사용 (cl100k_base = GPT-4 토크나이저)
            encoder = tiktoken.get_encoding("cl100k_base")
            self._encoders[model_name] = encoder

            return encoder
    
    def count_tokens(self, text: str, model_name: str = "gemini-2.5-flash") -> int:
        """텍스트의 토큰 수 계산"""
        try:
            encoder = self._get_encoder(model_name)
            tokens = encoder.encode(text)

            return len(tokens)

        except Exception as e:
            self.logger.error(f"토큰 계산 실패: {e}")

            # 대략적인 계산 (1 토큰 ≈ 4글자)
            return len(text) // 4
    
    def estimate_tokens_simple(self, text: str) -> int:
        """간단한 토큰 추정 (tiktoken 없이)"""
        # 영어: 1 토큰 ≈ 4글자
        # 한국어: 1 토큰 ≈ 1-2글자
        korean_chars = sum(1 for char in text if ord(char) >= 0xAC00 and ord(char) <= 0xD7AF)
        other_chars = len(text) - korean_chars
        
        # 한국어는 1.5글자당 1토큰, 영어는 4글자당 1토큰으로 추정
        estimated_tokens = korean_chars // 1.5 + other_chars // 4

        return int(estimated_tokens)
    
    def check_token_limits(self, text: str, model_name: str = "gemini-2.5-flash") -> dict:
        """토큰 제한 확인"""
        token_count = self.count_tokens(text, model_name)
        limits = ModelLimits.get_model_limits(model_name)
        
        return {
            "token_count": token_count,
            "model_limits": limits,
            "input_ok": token_count <= limits.get("max_input_tokens", 4096),
            "context_usage_percent": (token_count / limits.get("context_window", 4096)) * 100,
            "remaining_tokens": limits.get("max_input_tokens", 4096) - token_count
        }
    
    def split_text_by_tokens(self, text: str, max_tokens: int, model_name: str = "gemini-2.5-flash") -> list[str]:
        """텍스트를 토큰 제한에 맞게 분할"""
        encoder = self._get_encoder(model_name)
        tokens = encoder.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks


def create_token_analyzer():
    """토큰 분석기 인스턴스 생성"""
    return TokenCounter()


def analyze_text_tokens(text: str, model_name: str = "gemini-2.5-flash") -> dict:
    """텍스트 토큰 분석 (편의 함수)"""
    counter = TokenCounter()

    return counter.check_token_limits(text, model_name)


def get_gemini_limits(model_name: str = "gemini-2.5-flash") -> dict:
    """Gemini 모델의 제한 정보 반환"""
    # 모델명을 정규화 (소문자, 하이픈 통일)
    model_name = model_name.lower().replace("_", "-")
    
    # 모델별 제한 정보 매핑
    model_limits = {
        "gemini-2.5-flash": ModelLimits.GEMINI_2_5_FLASH,
        "gemini-2.5-pro": ModelLimits.GEMINI_2_5_PRO,
        "gemini-2.0-flash": ModelLimits.GEMINI_2_0_FLASH,
        "gemini-1.5-flash": ModelLimits.GEMINI_1_5_FLASH,
        "gemini-1.5-pro": ModelLimits.GEMINI_1_5_PRO,
    }
    
    # 모델명이 일치하면 해당 제한 정보 반환
    for model_key, limits in model_limits.items():
        if model_key in model_name:
            return limits
    
    # 기본값으로 Gemini 2.5 Flash 반환
    logger.warning(f"알 수 없는 Gemini 모델: {model_name}. 기본값(Gemini 2.5 Flash) 사용")

    return ModelLimits.GEMINI_2_5_FLASH


# 사용 예제 함수들
def demo_token_counting():
    """토큰 계산 데모"""
    counter = TokenCounter()
    
    # 테스트 텍스트들
    texts = [
        "안녕하세요! Gemini 2.0 Flash 모델을 사용하고 있습니다.",
        "Hello! I'm using Gemini 2.0 Flash model for my AI applications.",
        "이것은 매우 긴 텍스트입니다. " * 100,  # 긴 텍스트
    ]
    
    print("🔍 토큰 계산 데모")
    print("=" * 60)
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. 텍스트: {text[:50]}...")
        
        # 토큰 계산
        result = counter.check_token_limits(text, "gemini-2.0-flash")
        
        print(f"   📊 토큰 수: {result['token_count']:,}")
        print(f"   📈 컨텍스트 사용률: {result['context_usage_percent']:.2f}%")
        print(f"   ✅ 입력 가능: {'예' if result['input_ok'] else '아니오'}")
        print(f"   🔄 남은 토큰: {result['remaining_tokens']:,}")


if __name__ == "__main__":
    demo_token_counting()