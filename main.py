import argparse
import os
import asyncio
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# PyTorch CUDA 메모리 최적화 설정 (모델 로드 전에 설정해야 함)
# expandable_segments: 파편화 방지
# max_split_size_mb: 256MB 이상 블록 분할 방지 (큰 연속 메모리 확보)
# garbage_collection_threshold: 60% 사용 시 자동 메모리 정리
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6'
# Tokenizer parallelism 경고 방지 (fork 전에 설정 필수)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from LORIN.llm.chatbot import Chatbot, LLMProvider
from LORIN.logger.logger import Logger, get_logger
from LORIN.process.base import main_process
from stage1_filtering.stage1_process import (
    DEFAULT_THRESHOLD,
    DEFAULT_TEACHER_CKPT,
    DEFAULT_STUDENT_ROOT,
    run_pipeline as run_stage1_pipeline,
)

# Load environment variables from .env file
load_dotenv()

# Logger 클래스 설정
try:
    logger_configurator = Logger(name="LORIN")
except Warning:
    pass


# ─────────────────────────────────────────────────────────
# Dense 벡터 인덱스 로더 (BGE + FAISS)
# ─────────────────────────────────────────────────────────
def build_vectorstore():
    """
    FAISS 검색 엔진 인덱스 경로를 준비하여 리트리버에 제공
    """
    logger = get_logger(__name__)

    # FAISS 검색 엔진의 인덱스 경로 사용
    index_path = Path(os.getenv("FAISS_INDEX_PATH", "LORIN/make_faiss/log_faiss_index_bge_gemma2"))

    # FAISS 인덱스 파일 존재 확인
    faiss_index_file = index_path / "index.faiss"

    if not faiss_index_file.exists():
        logger.warning(
            "FAISS index file not found at %s. Retrieval will return empty results until the index is prepared.",
            faiss_index_file,
        )
    else:
        logger.info("FAISS index ready at %s", faiss_index_file)

    # 리트리버는 경로 문자열을 활용해 FAISS 검색 엔진을 초기화한다.
    return str(index_path), index_path


def _build_stage1_args(
    log_file: Path,
    output_dir: Optional[Path],
    anomaly_out: Optional[Path],
    faiss_dir: Optional[Path],
) -> argparse.Namespace:
    """Assemble args for stage1_process.run_pipeline with FAISS build enabled."""
    return argparse.Namespace(
        log_file=log_file,
        output_dir=output_dir,
        anomaly_out=anomaly_out,
        threshold=DEFAULT_THRESHOLD,
        embed_model="sentence-transformers/all-mpnet-base-v2",
        sbert_batch=64,
        window_size=50,
        stride=50,
        batch=192,
        teacher_ckpt=DEFAULT_TEACHER_CKPT,
        student_root=DEFAULT_STUDENT_ROOT,
        student_ckpt_files="",
        seeds="42,77,99",
        ckpts="best",
        build_faiss=True,
        faiss_dir=faiss_dir,
        faiss_batch=4,
        run_main=False,
    )


def run_stage1_and_prepare_faiss(
    log_file: Path,
    *,
    output_dir: Optional[Path] = None,
    anomaly_out: Optional[Path] = None,
    faiss_dir: Optional[Path] = None,
) -> Path:
    """
    Execute stage1_process to label logs, build a FAISS index, and set FAISS_INDEX_PATH.
    """
    logger = get_logger(__name__)

    args = _build_stage1_args(
        log_file=log_file.expanduser().resolve(),
        output_dir=output_dir,
        anomaly_out=anomaly_out,
        faiss_dir=faiss_dir,
    )

    logger.info("[Stage1] Starting stage1_process with FAISS build")
    result = run_stage1_pipeline(args)

    if not result.faiss_index_dir:
        raise RuntimeError("stage1_process did not return a FAISS index directory")

    os.environ["FAISS_INDEX_PATH"] = str(result.faiss_index_dir)
    logger.info("[Stage1] FAISS index ready at %s", result.faiss_index_dir)
    return result.faiss_index_dir


# ─────────────────────────────────────────────────────────
# Dense retrieval 설정 - BM25/Sparse 코드 제거됨
# ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────
async def main(experiment_config: Optional[dict] = None):
    """메인 실행 함수

    Args:
        experiment_config: 실험용 설정 딕셔너리 (선택적)
            예: {"use_planner": True, "use_activity_detection": False, ...}
    """
    logger = get_logger(__name__)
    logger.debug("Service begins")

    if experiment_config:
        logger.info(f"Running with experiment configuration: {experiment_config}")

    # 챗봇 설정 (환경변수 기반 유연한 모델 선택)
    # 주의: route.py에서 각 노드별 독립 인스턴스를 생성합니다
    # 여기서는 설정 검증 및 로깅 목적으로만 인스턴스 생성
    try:
        # 환경변수에서 LLM provider와 model 읽기 (기본값: EXAONE)
        provider_str = os.getenv("LLM_PROVIDER", "exaone").upper()
        model_name = os.getenv("LLM_MODEL", "exaone-4.0.1-32b")

        # 문자열을 LLMProvider enum으로 변환
        try:
            provider = LLMProvider[provider_str]
        except KeyError:
            logger.error(f"Invalid LLM_PROVIDER: {provider_str}. Valid options: {', '.join([p.name for p in LLMProvider])}")
            raise ValueError(f"지원하지 않는 LLM 프로바이더: {provider_str}")

        # 설정 검증용 chatbot 인스턴스 생성
        # 실제 노드들은 route.py에서 각각 독립 인스턴스 생성
        chatbot = Chatbot(
            provider=provider,
            model=model_name,
            temperature=0.4
        )
        logger.info("=" * 80)
        logger.info("LLM Configuration:")
        logger.info(f"  Provider: {chatbot.provider.value}")
        logger.info(f"  Model: {chatbot.model}")
        logger.info(f"  Note: Each node will create its own independent instance")
        logger.info(f"  Parallel Processing: ENABLED (async/await + asyncio.gather)")
        logger.info("=" * 80)
    except (ValueError, KeyError) as e:
        logger.error(f"LLM configuration validation failed: {e}")
        return

    # 1) 벡터 인덱스 경로 준비 (BGE + FAISS)
    vectorstore, index_path = build_vectorstore()

    # 2) 메인 프로세스 호출 (실험 설정 전달)
    await main_process(
        chatbot,
        vectorstore,
        config=experiment_config  # 실험 설정 전달
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description="LORIN pipeline entrypoint. Optionally run Stage1 (parse+label+FAISS) before main flow."
    )
    cli.add_argument(
        "--log-file",
        type=Path,
        help="Raw .log file to run through Stage1 (Drain + anomaly detection + FAISS build).",
    )
    cli.add_argument(
        "--stage1-output-dir",
        type=Path,
        default=None,
        help="Parsing output directory (default: <log_dir>/result).",
    )
    cli.add_argument(
        "--stage1-anomaly-dir",
        type=Path,
        default=None,
        help="Anomaly output directory (default: <stage1-output-dir>/anomaly_detection).",
    )
    cli.add_argument(
        "--faiss-dir",
        type=Path,
        default=None,
        help="Directory to save the FAISS index (default: <stage1-anomaly-dir>/faiss_index).",
    )
    cli.add_argument(
        "--skip-main",
        action="store_true",
        help="Run Stage1 only (if log-file is provided) and skip the main flow.",
    )
    args = cli.parse_args()

    if args.log_file:
        run_stage1_and_prepare_faiss(
            args.log_file,
            output_dir=args.stage1_output_dir,
            anomaly_out=args.stage1_anomaly_dir,
            faiss_dir=args.faiss_dir,
        )

    if not args.skip_main:
        asyncio.run(main())
