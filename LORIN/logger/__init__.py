"""
CaucowChat Logger Module - Advanced Logging System
==================================================

고급 로깅 기능을 제공하는 CaucowChat의 핵심 로깅 모듈입니다.
파일 로깅, 콘솔 로깅, 로그 회전 등 엔터프라이즈급 로깅 기능을 제공합니다.

주요 구성 요소
-----------
- **Logger**: 메인 로거 설정 및 관리 클래스
- **get_logger**: 모듈별 로거 인스턴스 생성 함수
- **LogLevel**: 로그 레벨 상수 정의

핵심 특징
--------
- 🗂️ **다중 출력**: 파일과 콘솔 동시 로깅
- 🔄 **자동 회전**: 크기/시간 기반 로그 파일 회전
- 📊 **레벨 필터링**: 세밀한 로그 레벨 제어
- ⚡ **고성능**: 최적화된 로깅 성능
- 🔧 **유연한 설정**: 환경 변수 및 코드 기반 설정

사용 패턴
--------
```python
# 1. 애플리케이션 초기화 시 (main.py)
from caucowchat.logger import Logger, LogLevel

logger_config = Logger(
    name="caucowchat",
    console_enabled=True,
    file_level=LogLevel.INFO
)

# 2. 각 모듈에서 사용
from caucowchat.logger import get_logger

logger = get_logger(__name__)
logger.info("모듈이 시작되었습니다")
```

로그 레벨 가이드
--------------
- **DEBUG**: 개발 및 디버깅용 상세 정보
- **INFO**: 일반적인 프로그램 흐름 정보
- **WARNING**: 주의가 필요하지만 프로그램은 계속 실행
- **ERROR**: 오류 발생, 일부 기능 실패
- **CRITICAL**: 심각한 오류, 프로그램 중단 가능성

모듈 구조
--------
```
logger/
├── logger.py     # 로거 구현체
└── __init__.py  # 공개 API
```

Dependencies
-----------
- logging: Python 표준 로깅
- pathlib: 경로 처리
- datetime: 타임스탬프
- python-dotenv: 환경 변수
"""

from .logger import Logger, get_logger, LogLevel

__all__ = [
    "Logger",
    "get_logger", 
    "LogLevel"
]
