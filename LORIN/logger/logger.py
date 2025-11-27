"""
Advanced Logger Module - Comprehensive Logging System
====================================================

CaucowChat을 위한 고급 로깅 시스템을 제공하는 모듈입니다.
파일 로깅, 콘솔 로깅, 로그 회전, 레벨별 필터링 등 다양한 기능을 지원합니다.

주요 클래스
----------
- Logger: 메인 로거 설정 및 관리 클래스
- LogLevel: 로그 레벨 상수 정의

로그 레벨
--------
- **DEBUG**: 상세한 진단 정보 (개발용)
- **INFO**: 일반적인 정보 메시지
- **WARNING**: 주의가 필요한 상황
- **ERROR**: 오류 발생 상황
- **CRITICAL**: 치명적인 오류

파일 구조
--------
```
logs/
├── default_YYYYMMDD.log      # 일반 로그
├── default_error_YYYYMMDD.log # 에러 로그
└── rotated/                  # rotate된 로그 파일들
```

환경 변수 설정
------------
```bash
# .env 파일에 설정
LOG_LEVEL=INFO
LOG_FILE_ENABLED=true
LOG_CONSOLE_ENABLED=true
LOG_ROTATION_ENABLED=true
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5
```

"""

import logging
import logging.handlers
from pathlib import Path
import os
from datetime import datetime
from dotenv import load_dotenv

# ── 중앙 집중식 설정 로드 ───────────────────────────────────
try:
    from ..config.settings import get_extended_logger_settings, get_settings
    _ext_logger_cfg = get_extended_logger_settings()
    _logger_cfg = get_settings().logger
    _LOG_MAX_BYTES = _ext_logger_cfg.max_bytes
    _LOG_BACKUP_COUNT = _ext_logger_cfg.backup_count
    _LOG_LEVEL = _logger_cfg.log_level
except ImportError:
    # settings 모듈이 없는 경우 환경변수/기본값 사용
    _LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
    _LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    _LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class LogLevel:
    """로그 레벨 상수 정의 클래스"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


_root_logger_configured = False # 루트 로거가 설정되었는지 추적하는 전역 플래그

class Logger:
    """
    주요 로깅 핸들러와 포맷터를 설정하는 클래스입니다.
    이 클래스는 애플리케이션 시작 시 단 한 번 인스턴스화되어야 합니다.
    """

    def __init__(
            self,
            name: str = 'default', # 이 이름은 로그 파일 접두사로 사용됩니다.
            file_level: str | int = None,
            console_level: str | int = None,
            console_enabled: bool = None,
            log_dir: str = 'logs',
            max_file_size: int = _LOG_MAX_BYTES,  # 중앙 설정에서 로드
            backup_count: int = _LOG_BACKUP_COUNT,  # 중앙 설정에서 로드
            daily_backup_count: int = 30
    ):
        global _root_logger_configured

        if _root_logger_configured:
            # 이미 설정되었다면 다시 설정하지 않음 (싱글턴 패턴)
            logging.warning("Logger 클래스가 이미 인스턴스화되었습니다. 여러 번 초기화하지 마십시오.")
            return

        # .env 파일 로드
        load_dotenv()

        self.name = name
        self.log_dir = Path(log_dir)
        self.file_level = self._get_level(file_level or os.getenv("LOG_FILE_LEVEL", "DEBUG"))
        self.console_level = self._get_level(console_level or os.getenv("LOG_CONSOLE_LEVEL", "INFO"))
        self.console_enabled = self._get_bool(console_enabled, "LOG_CONSOLE_ENABLED", False)

        # 로테이션 설정
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.daily_backup_count = daily_backup_count

        # 루트 로거 가져오기 및 설정
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)  # 모든 메시지가 핸들러로 전달되도록 가장 낮은 레벨로 설정

        # 기존 핸들러 제거 (중복 방지)
        self.root_logger.handlers.clear()

        # 로그 디렉터리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 파일 핸들러들 추가
        self._add_file_handlers(self.root_logger)

        # 콘솔 핸들러 추가 (플래그에 따라)
        if self.console_enabled:
            self._add_console_handler(self.root_logger)

        _root_logger_configured = True # 설정 완료 플래그

        # 초기화 완료 메시지는 외부에서 get_logger를 통해 로그하도록 유도
        logging.debug(f"Logger initialized (default name: {name})")
        logging.debug(f"Initial configuration: {self.get_config()}")


    def _get_bool(self, value: bool | None, env_key: str, default: bool) -> bool:
        """불린 값 결정 (인수 -> 환경변수 -> 기본값)"""
        if value is not None:
            return value
        env_value = os.getenv(env_key, "").lower()
        if env_value in ("true", "1", "yes", "on"):
            return True
        elif env_value in ("false", "0", "no", "off"):
            return False
        return default

    def _get_level(self, level: str | int) -> int:
        """문자열 또는 숫자 레벨을 숫자로 변환"""
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.INFO)
        return level

    def _add_file_handlers(self, target_logger: logging.Logger):
        """파일 핸들러들 추가"""

        # 포맷터 설정
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. 메인 로그 파일 (모든 로그)
        self._add_main_log_handler(target_logger, detailed_formatter)

        # 2. 레벨별 분리 파일들
        self._add_level_separated_handlers(target_logger, detailed_formatter)

        # 3. 일별 로그 파일
        self._add_daily_handler(target_logger, simple_formatter)

    def _add_main_log_handler(self, target_logger: logging.Logger, formatter: logging.Formatter):
        """메인 로그 파일 핸들러 추가"""

        log_file = self.log_dir / f"{self.name}.log"

        # 크기 기반 로테이션
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )

        handler.setLevel(self.file_level)
        handler.setFormatter(formatter)
        target_logger.addHandler(handler)

    def _add_level_separated_handlers(self, target_logger: logging.Logger, formatter: logging.Formatter):
        """레벨별 분리 파일 핸들러들 추가"""

        level_configs = [
            ("error", logging.ERROR, "ERROR와 CRITICAL"),
            ("warning", logging.WARNING, "WARNING 이상"),
            ("info", logging.INFO, "INFO 이상")
        ]

        for level_name, level_value, description in level_configs:
            log_file = self.log_dir / f"{self.name}_{level_name}.log"

            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size // 2,  # 레벨별 파일은 절반 크기
                backupCount=self.backup_count,
                encoding='utf-8'
            )

            handler.setLevel(level_value)
            handler.setFormatter(formatter)
            target_logger.addHandler(handler)

    def _add_daily_handler(self, target_logger: logging.Logger, formatter: logging.Formatter):
        """일별 로테이션 핸들러 추가"""

        log_file = self.log_dir / f"{self.name}_daily.log"

        handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=self.daily_backup_count,
            encoding='utf-8'
        )
        handler.setLevel(self.file_level)
        handler.setFormatter(formatter)

        # 파일명에 날짜 포함
        handler.suffix = "%Y%m%d"

        target_logger.addHandler(handler)

    def _add_console_handler(self, target_logger: logging.Logger):
        """콘솔 핸들러 추가"""

        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(console_formatter)

        target_logger.addHandler(console_handler)

    def enable_console(self, level: str | int = None):
        """콘솔 출력 활성화"""
        if not self.console_enabled:
            self.console_enabled = True
            if level is not None:
                self.console_level = self._get_level(level)
            # 루트 로거의 핸들러를 업데이트
            for handler in self.root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(self.console_level)
                    break
            else: # 기존 StreamHandler가 없으면 새로 추가
                self._add_console_handler(self.root_logger)
            logging.info("콘솔 출력이 활성화되었습니다")

    def disable_console(self):
        """콘솔 출력 비활성화"""
        if self.console_enabled:
            self.console_enabled = False
            # 콘솔 핸들러 제거
            self.root_logger.handlers = [
                h for h in self.root_logger.handlers
                if not isinstance(h, logging.StreamHandler)
            ]
            # 마지막 메시지는 파일에만 기록됨
            logging.info("콘솔 출력이 비활성화되었습니다")

    def set_file_level(self, level: str | int):
        """파일 로그 레벨 변경"""
        old_level = logging.getLevelName(self.file_level)
        self.file_level = self._get_level(level)
        new_level = logging.getLevelName(self.file_level)

        # 파일 핸들러들의 레벨 업데이트
        for handler in self.root_logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                handler.setLevel(self.file_level)

        logging.info(f"파일 로그 레벨 변경: {old_level} -> {new_level}")

    def set_console_level(self, level: str | int):
        """콘솔 로그 레벨 변경"""
        old_level = logging.getLevelName(self.console_level)
        self.console_level = self._get_level(level)
        new_level = logging.getLevelName(self.console_level)

        # 콘솔 핸들러의 레벨 업데이트
        for handler in self.root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(self.console_level)

        logging.info(f"콘솔 로그 레벨 변경: {old_level} -> {new_level}")

    def get_config(self) -> dict[str, object]:
        """현재 설정 반환"""
        return {
            "name": self.name,
            "file_level": logging.getLevelName(self.file_level),
            "console_level": logging.getLevelName(self.console_level),
            "console_enabled": self.console_enabled,
            "log_dir": str(self.log_dir),
            "max_file_size_mb": self.max_file_size // (1024 * 1024),
            "backup_count": self.backup_count,
            "daily_backup_count": self.daily_backup_count,
            "handlers_count": len(self.root_logger.handlers)
        }

    def get_log_files(self) -> dict[str, list]:
        """현재 생성된 로그 파일들 목록 반환"""
        files = {
            "main": [],
            "level_separated": [],
            "daily": [],
            "other": []
        }

        if not self.log_dir.exists():
            return files

        for file_path in self.log_dir.glob("*.log*"):
            file_name = file_path.name
            file_info = {
                "name": file_name,
                "size_mb": file_path.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }

            if file_name.startswith(f"{self.name}_error") or \
                    file_name.startswith(f"{self.name}_warning") or \
                    file_name.startswith(f"{self.name}_info"):
                files["level_separated"].append(file_info)
            elif file_name.startswith(f"{self.name}_daily"):
                files["daily"].append(file_info)
            elif file_name.startswith(f"{self.name}.log"):
                files["main"].append(file_info)
            else:
                files["other"].append(file_info)

        return files


class ColoredFormatter(logging.Formatter):
    """컬러가 있는 로그 포맷터 (콘솔용)"""

    COLORS = {
        'DEBUG': '\033[36m',  # 청록색
        'INFO': '\033[32m',  # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',  # 빨간색
        'CRITICAL': '\033[35m',  # 자주색
    }
    RESET = '\033[0m'

    def format(self, record):
        if hasattr(record, 'levelname'):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """
    주어진 이름의 표준 로거 인스턴스를 가져옵니다.
    이 로거는 루트 로거의 설정을 상속받습니다.
    """
    return logging.getLogger(name)