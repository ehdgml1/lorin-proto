# gui/parse_log.py
# gui/parse_log.py
"""
Drain 알고리즘 기반 로그 파서 래퍼.

Android 로그 텍스트를 Logparse/Drain.py의 `LogParser`를 활용해 구조화한다.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional, Sequence, Tuple

import pandas as pd
import re
import os

import sys

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.append(str(MODULE_ROOT))

from Logparse.Drain import LogParser

# ── 중앙 집중식 설정 로드 ───────────────────────────────────
try:
    from ..config.settings import get_drain_settings
    _drain_cfg = get_drain_settings()
    DEFAULT_THRESHOLD = _drain_cfg.default_threshold
except ImportError:
    # settings 모듈이 없는 경우 환경변수/기본값 사용
    DEFAULT_THRESHOLD = int(os.getenv("DRAIN_DEFAULT_THRESHOLD", "5"))

DEFAULT_LOG_NAME = "Android_Log.txt"
DEFAULT_LOG_FORMAT = "<Date> <Time>  <Level>/<Component>(<Pid>): <Content>"
DEFAULT_REGEX = [
    r"(/[\w-]+)+",  # 파일 경로
    r"([\w-]+\.){2,}[\w-]+",  # 도메인 형식
    r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",  # 숫자/16진수
]
DEFAULT_DELIMETER: list[str] = []
RESULT_DIR_NAME = "brain_result"
THREADTIME_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<date>\d{2}-\d{2})\s+
    (?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+
    (?P<pid>\d+)\s+
    (?P<tid>\d+)\s+
    (?P<level>[A-Z])\s+
    (?P<tag>[^:]+)\s*:\s*
    (?P<message>.*)
    $
    """,
    re.VERBOSE,
)


def _run_brain_parser(log_path: Path, outdir: Optional[Path] = None) -> pd.DataFrame:
    """
    주어진 로그 파일 경로를 Drain LogParser로 파싱한 뒤 DataFrame을 반환한다.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {log_path}")

    result_dir = outdir or log_path.parent / RESULT_DIR_NAME
    result_dir.mkdir(parents=True, exist_ok=True)

    parser = LogParser(
        logname="Android",
        log_format=DEFAULT_LOG_FORMAT,
        indir=str(log_path.parent),
        outdir=str(result_dir),
        threshold=DEFAULT_THRESHOLD,
        delimeter=DEFAULT_DELIMETER,
        rex=DEFAULT_REGEX,
    )

    parser.parse(log_path.name)
    return parser.df_log.copy()


def _extract_threadtime_rows(text: str) -> list[dict[str, Optional[object]]]:
    rows: list[dict[str, Optional[object]]] = []
    for line in text.splitlines():
        match = THREADTIME_PATTERN.match(line)
        if match:
            try:
                pid = int(match.group("pid"))
            except ValueError:
                pid = None
            try:
                tid = int(match.group("tid"))
            except ValueError:
                tid = None
            rows.append(
                {
                    "Date": match.group("date"),
                    "Time": match.group("time"),
                    "Level": match.group("level"),
                    "Component": match.group("tag").strip(),
                    "Pid": pid,
                    "Tid": tid,
                    "Content": match.group("message").strip(),
                }
            )
    return rows


def _attach_tid_column(
    df: pd.DataFrame, rows: Sequence[dict[str, Optional[object]]]
) -> pd.DataFrame:
    if df.empty:
        if rows:
            fallback_df = pd.DataFrame(rows)
            fallback_df.insert(0, "LineId", range(1, len(fallback_df) + 1))
            return fallback_df
        return df

    if rows:
        tids = [row.get("Tid") for row in rows]
        padded = list(tids) + [None] * max(0, len(df) - len(rows))
        trimmed = padded[: len(df)]
        df = df.copy()
        insert_at = df.columns.get_loc("Pid") + 1 if "Pid" in df.columns else len(df.columns)
        df.insert(insert_at, "Tid", trimmed)
    else:
        if "Tid" not in df.columns:
            df = df.copy()
            insert_at = df.columns.get_loc("Pid") + 1 if "Pid" in df.columns else len(df.columns)
            df.insert(insert_at, "Tid", [None] * len(df))
    return df


def parse_file(path: Path) -> pd.DataFrame:
    """지정한 파일 경로를 파싱한다."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = _extract_threadtime_rows(text)
    df = _run_brain_parser(path)
    return _attach_tid_column(df, rows)


def parse_text(text: str) -> pd.DataFrame:
    """
    업로드된 텍스트를 임시 파일로 저장한 뒤 Drain 파서를 실행한다.
    """
    if not text.strip():
        return pd.DataFrame(
            columns=["LineId", "Date", "Time", "Level", "Component", "Pid", "Content"]
        )

    with NamedTemporaryFile("w+", suffix=".log", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp.flush()
        temp_path = Path(tmp.name)

    try:
        with TemporaryDirectory() as tmpdir:
            df = _run_brain_parser(temp_path, Path(tmpdir))
    finally:
        temp_path.unlink(missing_ok=True)

    rows = _extract_threadtime_rows(text)
    return _attach_tid_column(df, rows)


def parse_default_log() -> pd.DataFrame:
    """
    GUI/Android_Log.txt 파일을 Drain 파서로 파싱해 DataFrame을 반환한다.
    """
    log_path = Path(__file__).with_name(DEFAULT_LOG_NAME)
    return parse_file(log_path)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """데이터프레임을 UTF-8 CSV 바이트로 변환한다."""
    return df.to_csv(index=False).encode("utf-8")


def _detect_log_format(text: str) -> str:
    """
    로그 형식을 감지합니다.
    
    Returns:
        "threadtime" 또는 "standard"
    """
    lines = text.splitlines()
    threadtime_count = 0
    standard_count = 0
    
    for line in lines[:100]:  # 처음 100줄만 확인
        line = line.strip()
        if not line:
            continue
        
        # threadtime 형식: MM-DD HH:MM:SS.mmm PID TID L Component: Content
        if THREADTIME_PATTERN.match(line):
            threadtime_count += 1
        # standard 형식: MM-DD HH:MM:SS.mmm L/Component(PID): Content
        elif re.match(r'^\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+[VDIWEFS]/', line):
            standard_count += 1
    
    if threadtime_count > standard_count:
        return "threadtime"
    else:
        return "standard"


def parse_text_to_csv_files(text: str, output_dir: Path) -> Tuple[Path, Path]:
    """
    업로드된 텍스트를 파싱하여 structured.csv와 templates.csv 파일을 생성한다.
    
    Args:
        text: 로그 텍스트
        output_dir: CSV 파일을 저장할 디렉토리
        
    Returns:
        tuple[Path, Path]: (structured_csv_path, templates_csv_path)
    """
    if not text.strip():
        raise ValueError("빈 텍스트는 파싱할 수 없습니다.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 형식 감지
    log_format = _detect_log_format(text)
    
    with NamedTemporaryFile("w+", suffix=".log", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp.flush()
        temp_path = Path(tmp.name)
    
    try:
        # threadtime 형식인 경우 threadtime 파서 사용
        if log_format == "threadtime":
            # threadtime 형식으로 파싱
            rows = _extract_threadtime_rows(text)
            if not rows:
                raise ValueError("threadtime 형식으로 파싱할 수 있는 로그가 없습니다.")
            
            # DataFrame 생성
            df = pd.DataFrame(rows)
            df.insert(0, "LineId", range(1, len(df) + 1))

            # Drain 파서로 템플릿 생성 (Content 컬럼만 사용)
            parser = LogParser(
                logname="Android",
                log_format=DEFAULT_LOG_FORMAT,
                indir=str(temp_path.parent),
                outdir=str(output_dir),
                threshold=DEFAULT_THRESHOLD,
                delimeter=DEFAULT_DELIMETER,
                rex=DEFAULT_REGEX,
            )
            
            # Content만 추출하여 임시 파일 생성
            with NamedTemporaryFile("w+", suffix=".log", delete=False, encoding="utf-8") as tmp_content:
                for content in df["Content"]:
                    # Drain 파서가 기대하는 형식으로 변환: MM-DD HH:MM:SS.mmm L/Component(PID): Content
                    # 실제로는 Content만 필요하므로 간단한 형식으로 변환
                    tmp_content.write(f"01-01 00:00:00.000 I/Component(0): {content}\n")
                tmp_content.flush()
                temp_content_path = Path(tmp_content.name)
            
            try:
                # Content만으로 템플릿 생성
                parser.parse(temp_content_path.name)
                
                # 파서 결과 확인
                if not hasattr(parser, 'df_log') or parser.df_log is None or len(parser.df_log) == 0:
                    raise ValueError("템플릿 생성에 실패했습니다.")
                
                # EventId와 EventTemplate 매핑 생성
                if "EventId" not in parser.df_log.columns or "EventTemplate" not in parser.df_log.columns:
                    raise ValueError("템플릿 생성 결과에 EventId 또는 EventTemplate이 없습니다.")
                
                # Content를 기준으로 EventId와 EventTemplate 매핑
                content_to_event = {}
                for idx, row in parser.df_log.iterrows():
                    content = row.get("Content", "")
                    event_id = row.get("EventId", "")
                    event_template = row.get("EventTemplate", "")
                    if content and event_id:
                        content_to_event[content] = (event_id, event_template)
                
                # 원본 DataFrame에 EventId와 EventTemplate 추가
                event_ids = []
                event_templates = []
                for content in df["Content"]:
                    if content in content_to_event:
                        event_id, event_template = content_to_event[content]
                        event_ids.append(event_id)
                        event_templates.append(event_template)
                    else:
                        # 매칭되지 않는 경우 기본값
                        event_ids.append("E0")
                        event_templates.append(content)
                
                df["EventId"] = event_ids
                df["EventTemplate"] = event_templates
                
            finally:
                temp_content_path.unlink(missing_ok=True)
            
            # structured CSV 저장
            structured_csv = output_dir / "Android_structured.csv"
            df.to_csv(structured_csv, index=False)
            
            # templates CSV 생성
            template_df = df.groupby(["EventId", "EventTemplate"]).size().reset_index(name="Occurrences")
            template_df = template_df[["EventId", "EventTemplate", "Occurrences"]]
            templates_csv = output_dir / "Android_templates.csv"
            template_df.to_csv(templates_csv, index=False)

        else:
            # standard 형식: Drain 파서 직접 사용
            parser = LogParser(
                logname="Android",
                log_format=DEFAULT_LOG_FORMAT,
                indir=str(temp_path.parent),
                outdir=str(output_dir),
                threshold=DEFAULT_THRESHOLD,
                delimeter=DEFAULT_DELIMETER,
                rex=DEFAULT_REGEX,
            )
            
            parser.parse(temp_path.name)
            
            # 파서의 df_log 확인
            if not hasattr(parser, 'df_log') or parser.df_log is None:
                raise ValueError("파서가 df_log를 생성하지 못했습니다.")
            
            if len(parser.df_log) == 0:
                raise ValueError("파서 결과가 비어있습니다. 로그 형식이 올바른지 확인하세요.")
            
            if "EventId" not in parser.df_log.columns:
                raise ValueError(f"파서 결과에 EventId 컬럼이 없습니다. 생성된 컬럼: {parser.df_log.columns.tolist()}")
            
            # 생성된 CSV 파일 경로
            structured_csv = output_dir / f"{parser.logName}_structured.csv"
            templates_csv = output_dir / f"{parser.logName}_templates.csv"
            
            # Tid 컬럼 추가 (structured.csv에)
            if structured_csv.exists():
                df = pd.read_csv(structured_csv, dtype={"EventId": str})
                rows = _extract_threadtime_rows(text)
                df = _attach_tid_column(df, rows)
                df.to_csv(structured_csv, index=False)
        
        # 파일 생성 확인
        if not structured_csv.exists():
            raise FileNotFoundError(f"structured CSV 파일이 생성되지 않았습니다: {structured_csv}")
        if not templates_csv.exists():
            raise FileNotFoundError(f"templates CSV 파일이 생성되지 않았습니다: {templates_csv}")
        
        # 최종 검증
        df = pd.read_csv(structured_csv, dtype={"EventId": str})
        if "EventId" not in df.columns:
            raise ValueError(f"EventId 컬럼이 생성되지 않았습니다. 생성된 컬럼: {df.columns.tolist()}")
        if "EventTemplate" not in df.columns:
            raise ValueError(f"EventTemplate 컬럼이 생성되지 않았습니다. 생성된 컬럼: {df.columns.tolist()}")
        
        tmpl_df = pd.read_csv(templates_csv, dtype={"EventId": str})
        if len(tmpl_df) == 0:
            raise ValueError("templates CSV가 비어있습니다. 로그 파싱이 제대로 되지 않았을 수 있습니다.")
        if "EventId" not in tmpl_df.columns:
            raise ValueError(f"templates CSV에 EventId 컬럼이 없습니다. 생성된 컬럼: {tmpl_df.columns.tolist()}")
        
        return structured_csv, templates_csv
    finally:
        temp_path.unlink(missing_ok=True)
