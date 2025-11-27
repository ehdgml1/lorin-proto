"""
CaucowChat Process Module - Unified Data Processing Framework
============================================================

Agent와 LLM을 통합한 고급 데이터 처리 프레임워크를 제공하는 모듈입니다.
단순한 텍스트 처리부터 복잡한 멀티스텝 워크플로우까지 다양한 처리 패턴을 지원합니다.

모듈 아키텍처
-----------
```
process/
├── base.py      # 프로세서 기본 클래스 및 구현체
└── __init__.py  # 공개 API 정의
```

"""

# Core classes
from .base import (
    main_process
)

from .route import (
    initialize_graph
)

__all__ = [
    "main_process",
    "initialize_graph"
]

__version__ = "0.1.0"
