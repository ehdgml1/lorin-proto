"""
LORIN Streamlit UI
로컬호스트 전용 간단한 UI
"""

import streamlit as st
import asyncio
from pathlib import Path
import sys
import pandas as pd
import os
import tempfile
import shutil
import atexit
import signal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit

# LORIN 모듈 import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

# 이상 탐지 모듈 import를 위한 경로 추가 (lorin-proto 내부의 stage1_filtering 디렉터리)
# 단, parse_log와의 충돌을 피하기 위해 나중에 추가
LORIN_PROTO_DIR = Path(__file__).parent.parent.parent
STAGE1_FILTERING_DIR = LORIN_PROTO_DIR / "stage1_filtering"

# 종료 시 GPU 메모리 정리 함수
def cleanup_gpu_on_exit():
    """프로세스 종료 시 GPU 메모리 정리"""
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 GPU 메모리 정리 완료 (프로세스 종료)")
    except Exception as e:
        print(f"GPU 정리 중 오류: {e}")

# 종료 핸들러 등록 (atexit만 사용 - Streamlit 호환)
atexit.register(cleanup_gpu_on_exit)

# Signal 핸들러는 메인 스레드에서만 작동하므로 조건부로 등록
try:
    def signal_handler(sig, frame):
        """Ctrl+C 시그널 처리"""
        print("\n🛑 종료 신호 수신 - GPU 메모리 정리 중...")
        cleanup_gpu_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except ValueError:
    # Streamlit에서는 signal이 메인 스레드가 아니므로 무시
    pass

# GPU 환경 설정 (build_all_faiss_indexes.py 패턴)
def setup_gpu_environment():
    """PyTorch GPU 메모리 관리 최적화"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"GPU 환경 설정 중 오류: {e}")

# 앱 시작 시 GPU 환경 최적화
setup_gpu_environment()

from LORIN.agent.state import create_initial_state, get_last_message
from LORIN.agent.graph import create_agent_graph
from LORIN.process.route import initialize_graph
from main import build_vectorstore
from LORIN.utils import create_chatbot_from_env
# LORIN/ui/parse_log.py를 명시적으로 import (stage1_filtering/parse_log.py와 구분)
# 상대 import를 사용하여 명시적으로 LORIN.ui.parse_log를 import
from LORIN.ui import parse_log as ui_parse_log
parse_text = ui_parse_log.parse_text
parse_text_to_csv_files = ui_parse_log.parse_text_to_csv_files
from io import StringIO
from contextlib import redirect_stdout
from types import SimpleNamespace

# 이상 탐지 모듈 import (lorin-proto 내부의 stage1_filtering 디렉터리에서)
# test.py가 models.model을 import하기 전에 stage1_filtering 디렉터리를 sys.path에 추가해야 함
ANOMALY_DETECTION_AVAILABLE = False
ANOMALY_DETECTION_ERROR = None
ANOMALY_DETECTION_ERROR_TRACEBACK = None
try:
    # test.py 내부에서 SCRIPT_DIR.parent를 sys.path에 추가하지만,
    # import 시에는 실행되지 않으므로 미리 추가
    # parse_log 충돌을 피하기 위해 나중에 추가
    if str(STAGE1_FILTERING_DIR) not in sys.path:
        sys.path.insert(0, str(STAGE1_FILTERING_DIR))

    from stage1_filtering.scripts.test import (
        build_dual_branch_and_save,
        infer_ensemble_unlabeled_and_label_rows,
    )
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError as e:
    ANOMALY_DETECTION_ERROR = str(e)
    import traceback
    ANOMALY_DETECTION_ERROR_TRACEBACK = traceback.format_exc()
except Exception as e:
    ANOMALY_DETECTION_ERROR = str(e)
    import traceback
    ANOMALY_DETECTION_ERROR_TRACEBACK = traceback.format_exc()

# 페이지 설정
st.set_page_config(
    page_title="LORIN Log Analyzer",
    page_icon="🔍",
    layout="wide"
)

# 화면 배경 하얀색으로 설정 + 요소들 어두운 색으로
st.markdown("""
    <style>
    /* 배경 */
    .stApp {
        background-color: white;
    }
    [data-testid="stAppViewContainer"] {
        background-color: white;
    }
    [data-testid="stHeader"] {
        background-color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* 텍스트 색상 */
    .stApp, .stMarkdown, p, span, label {
        color: #1a1a1a !important;
    }

    /* 헤더 텍스트 */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    /* 버튼 - 부드러운 크림 베이지 계열 */
    .stButton > button {
        background-color: #fceacd;
        color: #333333 !important;
        border: 2px solid #f5dbb8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #f5dbb8;
        border-color: #eecca3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(252, 234, 205, 0.3);
    }
    /* Primary 버튼 (type="primary") */
    .stButton > button[kind="primary"] {
        background-color: #fceacd;
        border-color: #f5dbb8;
        color: #333333 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #f5dbb8;
        border-color: #eecca3;
    }

    /* 입력창 */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #f8f9fa;
        color: #1a1a1a !important;
        border: 2px solid #dee2e6;
    }

    /* 선택박스 */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        color: #1a1a1a !important;
        border: 2px solid #dee2e6;
    }

    /* 체크박스 */
    .stCheckbox > label {
        color: #1a1a1a !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #e9ecef;
        color: #1a1a1a !important;
        font-weight: 600;
    }

    /* 경고/성공/에러 메시지 */
    .stAlert {
        color: #1a1a1a !important;
    }

    /* 코드 블록 */
    code {
        background-color: #f8f9fa;
        color: #c7254e;
        border: 1px solid #dee2e6;
    }

    /* 파일 업로더 */
    [data-testid="stFileUploader"] {
        background-color: #e8f4f8;
        border: 2px dashed #3498db;
        border-radius: 8px;
        padding: 20px;
    }
    [data-testid="stFileUploader"] label {
        color: #2c3e50 !important;
        font-weight: 600;
        font-size: 16px;
    }
    [data-testid="stFileUploader"] section {
        border: none;
    }
    [data-testid="stFileUploader"] section > div {
        color: #2c3e50 !important;
        font-weight: 500;
        font-size: 14px;
    }
    /* Drag and drop 텍스트 */
    [data-testid="stFileUploader"] small {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    [data-testid="stFileUploader"] p {
        color: #2c3e50 !important;
    }
    /* 업로드 버튼 */
    [data-testid="stFileUploader"] button {
        background-color: #3498db;
        color: white !important;
        border: none;
        border-radius: 4px;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #2980b9;
    }

    /* 분석 중 애니메이션 */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .analyzing-emoji {
        display: inline-block;
        animation: spin 2s linear infinite;
        font-size: 1.2em;
        margin-right: 8px;
    }
    .progress-text {
        font-weight: 600;
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# 임베딩 모델 캐싱 (build_all_faiss_indexes.py 패턴)
@st.cache_resource
def get_embeddings_model(batch_size: int = 8):
    """
    임베딩 모델을 캐싱하여 재사용 (build_all_faiss_indexes.py 패턴)

    Args:
        batch_size: 배치 크기 (기본값: 8 - balanced mode)

    Returns:
        BGEGemma2MultiGPUEmbeddings: 캐싱된 임베딩 모델
    """
    from LORIN.make_faiss.make_faiss_with_bge_multilingual_gemma2 import (
        BGEGemma2MultiGPUEmbeddings
    )

    st.info(f"🤖 임베딩 모델 로딩 중 (batch_size={batch_size})...")
    embeddings = BGEGemma2MultiGPUEmbeddings(
        model_name="BAAI/bge-multilingual-gemma2",
        use_multi_gpu=False,
        batch_size_per_gpu=batch_size,
        use_fp16=True
    )
    st.success("✅ 임베딩 모델 로드 완료 (캐싱됨)")
    return embeddings

# DataFrame에서 FAISS 인덱스 빌드 함수
def build_faiss_from_df(df: pd.DataFrame, output_dir: Path):
    """
    DataFrame으로부터 FAISS 인덱스 빌드

    Args:
        df: 구조화된 로그 DataFrame
        output_dir: 인덱스 저장 디렉토리

    Returns:
        str: 빌드된 인덱스 경로
    """
    import torch
    import gc
    from LORIN.make_faiss.make_faiss_with_bge_multilingual_gemma2 import build_docs_from_df
    from langchain_community.vectorstores import FAISS

    # 문서 생성
    try:
        with st.spinner("📄 문서 청크 생성 중..."):
            docs = build_docs_from_df(df)
            st.success(f"✅ {len(docs):,} 문서 청크 생성 완료")
    except Exception as e:
        st.error(f"❌ 문서 생성 실패: {e}")
        return None

    # 캐싱된 임베딩 모델 가져오기 (build_all_faiss_indexes.py 패턴)
    embeddings = get_embeddings_model(batch_size=4)

    # FAISS 인덱스 빌드 (build_single_index 패턴)
    try:
        with st.spinner("🔨 FAISS 인덱스 빌드 중..."):
            vectorstore = FAISS.from_documents(docs, embeddings)

            # 인덱스 저장
            output_dir.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(output_dir))

            # 파일 크기 확인
            faiss_file = output_dir / "index.faiss"
            faiss_size = faiss_file.stat().st_size / (1024**2) if faiss_file.exists() else 0
            st.success(f"✅ FAISS 인덱스 빌드 완료 ({faiss_size:.1f} MB)")

    except torch.cuda.OutOfMemoryError as e:
        st.error(f"❌ GPU 메모리 부족: {e}")
        return None
    except Exception as e:
        st.error(f"❌ FAISS 인덱스 빌드 실패: {e}")
        return None
    finally:
        # 명시적 메모리 해제 (build_single_index 패턴)
        if 'vectorstore' in locals():
            del vectorstore
        if 'docs' in locals():
            del docs

        # 완전한 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # GPU 작업 완료 대기
            torch.cuda.empty_cache()   # GPU 캐시 비우기
            gc.collect()               # Python GC 실행

    return str(output_dir)

# TXT 파일 업로드 및 FAISS 인덱스 빌드 함수 (레거시 - 호환성 유지)
def build_faiss_from_txt(txt_file, output_dir: Path):
    """
    업로드된 TXT 로그 파일을 파싱하여 FAISS 인덱스 빌드

    Args:
        txt_file: Streamlit UploadedFile 객체 (TXT 파일)
        output_dir: 인덱스 저장 디렉토리

    Returns:
        str: 빌드된 인덱스 경로
    """
    import torch
    import gc
    from LORIN.make_faiss.make_faiss_with_bge_multilingual_gemma2 import build_docs_from_df
    from langchain_community.vectorstores import FAISS

    # TXT 파일 읽기 및 파싱
    try:
        # 파일 내용 읽기
        bytes_data = txt_file.read()
        try:
            text = bytes_data.decode("utf-8")
        except UnicodeDecodeError:
            text = bytes_data.decode("utf-8", errors="ignore")
        
        st.info(f"📄 TXT 파일 로드 완료: {len(text.splitlines()):,} lines")
        
        # 텍스트 파싱
        with st.spinner("📝 로그 파일 파싱 중..."):
            df = parse_text(text)
        st.info(f"📊 파싱 완료: {len(df):,} rows")
    except Exception as e:
        st.error(f"❌ 파일 로드/파싱 실패: {e}")
        import traceback
        with st.expander("상세 오류"):
            st.code(traceback.format_exc())
        return None

    # 문서 생성
    try:
        with st.spinner("📄 문서 청크 생성 중..."):
            docs = build_docs_from_df(df)
            st.success(f"✅ {len(docs):,} 문서 청크 생성 완료")
    except Exception as e:
        st.error(f"❌ 문서 생성 실패: {e}")
        return None

    # 캐싱된 임베딩 모델 가져오기 (build_all_faiss_indexes.py 패턴)
    embeddings = get_embeddings_model(batch_size=4)

    # FAISS 인덱스 빌드 (build_single_index 패턴)
    try:
        with st.spinner("🔨 FAISS 인덱스 빌드 중..."):
            vectorstore = FAISS.from_documents(docs, embeddings)

            # 인덱스 저장
            output_dir.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(output_dir))

            # 파일 크기 확인
            faiss_file = output_dir / "index.faiss"
            faiss_size = faiss_file.stat().st_size / (1024**2) if faiss_file.exists() else 0
            st.success(f"✅ FAISS 인덱스 빌드 완료 ({faiss_size:.1f} MB)")

    except torch.cuda.OutOfMemoryError as e:
        st.error(f"❌ GPU 메모리 부족: {e}")
        return None
    except Exception as e:
        st.error(f"❌ FAISS 인덱스 빌드 실패: {e}")
        return None
    finally:
        # 명시적 메모리 해제 (build_single_index 패턴)
        if 'vectorstore' in locals():
            del vectorstore
        if 'docs' in locals():
            del docs
        if 'df' in locals():
            del df
        # embeddings는 캐싱된 모델이므로 삭제하지 않음!

        # 완전한 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # GPU 작업 완료 대기
            torch.cuda.empty_cache()   # GPU 캐시 비우기
            gc.collect()               # Python GC 실행

    return str(output_dir)

# 싱글턴 초기화 (Streamlit cache 활용)
@st.cache_resource
def initialize_analyzer(index_path: str):
    """인덱스 경로가 제공된 경우에만 초기화

    Args:
        index_path: FAISS 인덱스 경로 (필수)
    """
    with st.spinner("🔄 시스템 초기화 중..."):
        # 커스텀 인덱스 경로 설정
        os.environ["FAISS_INDEX_PATH"] = index_path

        vectorstore, _ = build_vectorstore()

        # Chatbot
        chatbot = create_chatbot_from_env(temperature=0.4)

        # Graph 생성 및 컴파일
        graph = create_agent_graph()
        graph = asyncio.run(initialize_graph(graph, chatbot, vectorstore))
        compiled_graph = graph.compile()

        return compiled_graph, chatbot, vectorstore

# 시각화 함수
def create_visualization(csv_path: str) -> str:
    """
    업로드된 CSV로 시각화 생성

    Args:
        csv_path: CSV 파일 경로

    Returns:
        str: 생성된 이미지 파일 경로
    """
    sys.path.append(str(Path(__file__).parent.parent / "log_data"))
    from visualize_logs_v6_final import create_v6_final_dashboard

    # 임시 출력 파일 경로
    output_path = Path(tempfile.gettempdir()) / f"lorin_viz_{os.getpid()}.png"

    # 시각화 생성
    create_v6_final_dashboard(csv_path, output_path=str(output_path))

    return str(output_path)

# 세션 상태 초기화
if "custom_index_path" not in st.session_state:
    st.session_state.custom_index_path = None
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False
if "uploaded_csv_path" not in st.session_state:
    st.session_state.uploaded_csv_path = None
if "anomaly_detection_enabled" not in st.session_state:
    st.session_state.anomaly_detection_enabled = True
# 이상 탐지 파이프라인 기본 경로 설정 (lorin-proto 내부의 stage1_filtering 디렉터리 기준)
DEFAULT_TEACHER_CKPT = STAGE1_FILTERING_DIR / "teacher_pretrained_aosp.pt"
DEFAULT_STUDENT_ROOT = STAGE1_FILTERING_DIR / "revkd_out_v5_aosp"
DEFAULT_SEEDS = "42,77,99"
DEFAULT_CKPTS = "best"

if "pipeline_teacher_ckpt" not in st.session_state:
    st.session_state.pipeline_teacher_ckpt = str(DEFAULT_TEACHER_CKPT)
if "pipeline_student_root" not in st.session_state:
    st.session_state.pipeline_student_root = str(DEFAULT_STUDENT_ROOT)
if "pipeline_student_ckpt_files" not in st.session_state:
    st.session_state.pipeline_student_ckpt_files = ""
if "pipeline_seeds" not in st.session_state:
    st.session_state.pipeline_seeds = DEFAULT_SEEDS
if "pipeline_ckpts" not in st.session_state:
    st.session_state.pipeline_ckpts = DEFAULT_CKPTS

# 조건부 초기화 (인덱스가 있을 때만)
if st.session_state.custom_index_path:
    compiled_graph, chatbot, vectorstore = initialize_analyzer(st.session_state.custom_index_path)
    st.session_state.system_initialized = True
else:
    compiled_graph, chatbot, vectorstore = None, None, None
    st.session_state.system_initialized = False

# UI 헤더
st.title("🔍 LORIN Log Analyzer")
st.markdown("멀티에이전트 기반 로그 분석 시스템")

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # TXT 파일 업로드 및 FAISS 인덱스 빌드 (최상단에 배치)
    st.subheader("📂 TXT 원본 파일 업로드")

    uploaded_file = st.file_uploader(
        "로그 TXT 파일 업로드",
        type=["txt", "log"],
        help="TXT 로그 파일을 업로드하면 자동으로 파싱 후 FAISS 인덱스를 빌드합니다"
    )

    # 이상 탐지 설정 - 자동으로 경로 설정 (사용자에게 표시하지 않음)
    if not ANOMALY_DETECTION_AVAILABLE:
        st.error("❌ 이상 탐지 모듈을 사용할 수 없습니다. stage1_filtering 모듈을 확인하세요.")
        st.info(f"💡 lorin-proto 디렉터리 경로: {LORIN_PROTO_DIR}")
        st.info(f"💡 stage1_filtering 디렉터리 경로: {STAGE1_FILTERING_DIR}")
        st.info(f"💡 stage1_filtering 디렉터리 존재 여부: {STAGE1_FILTERING_DIR.exists()}")
        if ANOMALY_DETECTION_ERROR:
            st.error(f"오류 메시지: {ANOMALY_DETECTION_ERROR}")
            if ANOMALY_DETECTION_ERROR_TRACEBACK:
                with st.expander("상세 오류 정보"):
                    st.code(ANOMALY_DETECTION_ERROR_TRACEBACK)
        st.stop()
    
    # 자동으로 경로 설정 (사용자 입력 불필요, UI에 표시하지 않음)
    teacher_ckpt = str(DEFAULT_TEACHER_CKPT)
    student_root = str(DEFAULT_STUDENT_ROOT)
    student_ckpt_files = ""
    seeds = DEFAULT_SEEDS
    ckpts = DEFAULT_CKPTS

    if uploaded_file is not None:
        if st.button("🔨 FAISS 인덱스 빌드 & 적용", type="primary"):
            try:
                # TXT 파일 읽기
                uploaded_file.seek(0)
                bytes_data = uploaded_file.read()
                try:
                    text = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    text = bytes_data.decode("utf-8", errors="ignore")
                
                # 임시 디렉토리 설정
                temp_base_dir = Path(tempfile.gettempdir()) / f"lorin_{uploaded_file.name.replace('.txt', '').replace('.log', '')}"
                temp_base_dir.mkdir(parents=True, exist_ok=True)
                
                # 1단계: 파싱하여 structured.csv와 templates.csv 생성
                with st.spinner("📝 로그 파일 파싱 중..."):
                    structured_csv, templates_csv = parse_text_to_csv_files(text, temp_base_dir)
                    st.success(f"✅ 파싱 완료: {structured_csv.name}, {templates_csv.name}")
                
                # 2단계: 이상 탐지 수행 (필수)
                if not ANOMALY_DETECTION_AVAILABLE:
                    st.error("❌ 이상 탐지 모듈을 사용할 수 없습니다. stage1_filtering 모듈을 확인하세요.")
                    st.stop()
                
                # 이상 탐지 파이프라인 실행
                anomaly_out_dir = temp_base_dir / "anomaly_detection"
                anomaly_out_dir.mkdir(parents=True, exist_ok=True)
                
                # 설정 검증
                errors = []
                if not teacher_ckpt.strip():
                    errors.append("교사 모델 체크포인트 경로를 입력하세요.")
                elif not Path(teacher_ckpt).exists():
                    errors.append(f"교사 모델 체크포인트가 존재하지 않습니다: {teacher_ckpt}")
                
                if student_root.strip() and not Path(student_root).exists():
                    errors.append(f"학생 모델 루트가 존재하지 않습니다: {student_root}")
                
                if errors:
                    for msg in errors:
                        st.error(msg)
                    st.stop()
                
                # 이상 탐지 파이프라인 실행
                with st.spinner("🔍 이상 탐지 파이프라인 실행 중..."):
                    args = SimpleNamespace(
                        struct_csv=structured_csv,
                        template_csv=templates_csv,
                        out=anomaly_out_dir,
                        embed_model="sentence-transformers/all-mpnet-base-v2",
                        sbert_batch=64,
                        window_size=50,
                        stride=50,
                        teacher_ckpt=Path(teacher_ckpt),
                        batch=192,
                        student_root=Path(student_root) if student_root.strip() else None,
                        student_ckpt_files=student_ckpt_files,
                        seeds=seeds,
                        ckpts=ckpts,
                    )
                    
                    log_buffer = StringIO()
                    try:
                        with redirect_stdout(log_buffer):
                            N_dedup, orig_to_dedup_idx, win_starts = build_dual_branch_and_save(args)
                            infer_ensemble_unlabeled_and_label_rows(args, N_dedup, orig_to_dedup_idx, win_starts)
                        
                        # 결과 CSV 경로
                        labeled_csv = anomaly_out_dir / "Android.log_structured_clean.labeled.csv"
                        if not labeled_csv.exists():
                            st.error("❌ 이상 탐지 결과 CSV를 찾을 수 없습니다.")
                            with st.expander("파이프라인 로그"):
                                st.code(log_buffer.getvalue())
                            st.stop()
                        
                        final_csv_path = labeled_csv
                        st.success(f"✅ 이상 탐지 완료: {labeled_csv.name}")
                    except Exception as exc:
                        logs = log_buffer.getvalue()
                        st.error(f"❌ 이상 탐지 파이프라인 실행 중 오류: {exc}")
                        with st.expander("상세 오류"):
                            import traceback
                            st.code(logs + "\n" + traceback.format_exc())
                        st.stop()
                
                # 3단계: FAISS 인덱스 빌드
                temp_index_dir = Path(tempfile.gettempdir()) / "lorin_custom_index"
                if temp_index_dir.exists():
                    shutil.rmtree(temp_index_dir)
                
                with st.spinner("🔨 FAISS 인덱스 빌드 중..."):
                    # CSV 파일을 읽어서 DataFrame으로 변환
                    df = pd.read_csv(final_csv_path, dtype={"EventId": str})
                    
                    # FAISS 인덱스 빌드
                    index_path = build_faiss_from_df(df, temp_index_dir)
                
                # 세션 상태에 저장
                if index_path:
                    st.session_state.custom_index_path = index_path
                    st.session_state.uploaded_csv_path = str(final_csv_path)
                    
                    # 캐시 초기화해서 새 인덱스 사용
                    st.cache_resource.clear()
                    
                    st.success("✅ 인덱스 적용 완료! 페이지를 새로고침합니다...")
                    st.rerun()
                else:
                    st.error("❌ 인덱스 빌드 실패")

            except Exception as e:
                st.error(f"❌ 처리 실패: {str(e)}")
                import traceback
                with st.expander("상세 오류"):
                    st.code(traceback.format_exc())

    # 시스템 상태 표시 (간소화)
    st.divider()
    if st.session_state.system_initialized:
        st.success("✅ 시스템 준비 완료")
        st.caption(f"인덱스: `{Path(st.session_state.custom_index_path).name}`")

        if st.button("🔄 새 TXT 파일 업로드"):
            st.session_state.custom_index_path = None
            st.session_state.system_initialized = False
            st.session_state.uploaded_csv_path = None
            st.cache_resource.clear()
            st.rerun()

    # LLM 정보 (초기화된 경우에만)
    if chatbot:
        st.divider()
        st.info(f"""
        **LLM 정보**
        - Provider: {chatbot.provider.value}
        - Model: {chatbot.model}
        """)

    # 실험 설정 (선택사항)
    st.divider()
    with st.expander("🔬 실험 설정"):
        experiment_mode = st.selectbox(
            "분석 모드",
            [
                "Full LORIN",
                "w/o Replanner",
                "w/o Evaluator",
                "Custom"
            ]
        )

        if experiment_mode == "Full LORIN":
            config = {
                "skip_planner": False,
                "skip_quality_evaluator": False,
                "skip_replanner": False
            }
            st.info("✅ 모든 컴포넌트 활성화")

        elif experiment_mode == "w/o Replanner":
            config = {
                "skip_planner": False,
                "skip_quality_evaluator": False,
                "skip_replanner": True
            }
            st.info("✅ Replanner 제외")

        elif experiment_mode == "w/o Evaluator":
            config = {
                "skip_planner": False,
                "skip_quality_evaluator": True,
                "skip_replanner": True
            }
            st.info("✅ Evaluator, Replanner 제외")

        else:  # Custom
            skip_planner = st.checkbox("Planner 비활성화", value=False)
            skip_evaluator = st.checkbox("Quality Evaluator 비활성화", value=False)
            skip_replanner = st.checkbox("Replanner 비활성화", value=False)

            config = {
                "skip_planner": skip_planner,
                "skip_quality_evaluator": skip_evaluator,
                "skip_replanner": skip_replanner
            }

            # 활성화된 컴포넌트 표시
            active = []
            if not skip_planner:
                active.append("Planner")
            active.append("Retriever")
            if not skip_evaluator:
                active.append("Evaluator")
            if not skip_replanner:
                active.append("Replanner")

            st.caption(f"활성 컴포넌트: {' → '.join(active)}")

# 세션 상태 초기화
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# 메인 영역
if not st.session_state.system_initialized:
    # 시스템 미초기화 시 안내 메시지
    st.warning("⚠️ TXT 로그 파일을 업로드하고 FAISS 인덱스를 빌드한 후 분석을 시작할 수 있습니다.")
    st.info("👈 왼쪽 사이드바에서 TXT 로그 파일을 업로드하세요.")
    st.stop()

st.subheader("📝 질문 입력")

# 예제 질문
examples = [
    "The data passed between processes is too large and this is causing a failure. Please tell me which log range I should check for debugging.",
    "When does the Android system boot process start the DropBoxManager service?",
    "프로세스 간 데이터 전달이 너무 커서 실패가 발생합니다. 디버깅을 위해 어느 로그 범위를 확인해야 하나요?"
]

selected_example = st.selectbox(
    "예제 질문 선택 (선택사항)",
    ["직접 입력"] + examples,
    key="example_selector"
)

# 입력 칸 초기화 로직
if st.session_state.clear_input:
    st.session_state.question_input = ""
    st.session_state.clear_input = False

if selected_example == "직접 입력":
    question_input = st.text_area(
        "질문을 입력하세요",
        value=st.session_state.question_input,
        height=150,
        placeholder="로그 분석 질문을 입력하세요...",
        key="question_area"
    )
    # 사용자가 입력한 값을 session_state에 저장
    st.session_state.question_input = question_input
    question = question_input
else:
    question = st.text_area(
        "질문을 입력하세요",
        value=selected_example,
        height=150,
        key="question_area_example"
    )

analyze_button = st.button("🚀 분석 시작", type="primary", use_container_width=True)

# 결과 영역
st.divider()
result_container = st.container()

# 분석 실행
if analyze_button and question:
    # 분석 시작 시 입력 칸 초기화 플래그 설정
    st.session_state.clear_input = True

    with st.spinner("🔍 분석 실행 중... 잠시만 기다려주세요!"):
        try:
            # State 생성
            metadata = {
                "experiment_config": config
            }
            state = create_initial_state(question, metadata=metadata)

            # 스트리밍 분석 실행
            async def run_analysis():
                async for event in compiled_graph.astream_events(state, version="v2"):
                    # 진행 상황 표시 없이 이벤트만 처리
                    pass

                # 최종 state 반환
                final_state = await compiled_graph.ainvoke(state)
                return final_state

            # 분석 실행
            final_state = asyncio.run(run_analysis())

            # GPU 메모리 정리 (CUDA out of memory 방지)
            try:
                import torch
                import gc

                if torch.cuda.is_available():
                    # PyTorch GPU 캐시 정리
                    torch.cuda.empty_cache()
                    # Python 가비지 컬렉션
                    gc.collect()

                    # 메모리 정리 로그
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                    st.info(f"🧹 GPU 메모리 정리 완료 (할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB)")
            except Exception as cleanup_error:
                st.warning(f"메모리 정리 중 경고: {cleanup_error}")

            # 최종 답변 표시
            last_message = get_last_message(final_state)

            with result_container:
                st.success("✨ 분석 완료!")

                # 📍 [NEW] 로그 범위 시각화 (experiment_output 있는 경우)
                experiment_output = final_state.get("experiment_output")
                if experiment_output and experiment_output.get("total_log_range"):
                    st.subheader("📍 Log Range Overview")

                    total_range = experiment_output["total_log_range"]
                    rec_range = experiment_output.get("recommended_range")

                    col_range1, col_range2, col_range3 = st.columns([2, 2, 1])

                    with col_range1:
                        st.metric(
                            "🗂️ Total Log Range",
                            f"Lines {total_range['start']:,} ~ {total_range['end']:,}",
                            delta=f"{total_range['count']:,} lines total"
                        )

                    with col_range2:
                        if rec_range:
                            st.metric(
                                "🔴 Recommended Focus Range",
                                f"Lines {rec_range['start']:,} ~ {rec_range['end']:,}",
                                delta=f"{rec_range['count']:,} lines ({rec_range['coverage']})"
                            )
                        else:
                            st.metric("🔴 Recommended Focus Range", "Not specified", delta="0 lines")

                    with col_range3:
                        st.metric(
                            "📊 Retrieved",
                            f"{total_range['retrieved_lines']:,}",
                            delta=total_range['retrieved_coverage']
                        )

                    # 시각화: Progress bar로 전체 범위 내 추천 범위 표시
                    if rec_range:
                        st.markdown("#### 📊 Visual Log Range Map")

                        # 전체 범위 대비 추천 범위의 위치 계산
                        total_start = total_range['start']
                        total_end = total_range['end']
                        total_span = total_end - total_start

                        rec_start = rec_range['start']
                        rec_end = rec_range['end']

                        # 상대적 위치 계산 (0.0 ~ 1.0)
                        if total_span > 0:
                            before_ratio = (rec_start - total_start) / total_span
                            focus_ratio = (rec_end - rec_start) / total_span
                            after_ratio = (total_end - rec_end) / total_span
                        else:
                            before_ratio = 0.0
                            focus_ratio = 1.0
                            after_ratio = 0.0

                        # HTML/CSS를 사용한 커스텀 시각화
                        st.markdown(f"""
                        <div style="background: linear-gradient(to right,
                            #e0e0e0 0%,
                            #e0e0e0 {before_ratio*100}%,
                            #ff4444 {before_ratio*100}%,
                            #ff4444 {(before_ratio+focus_ratio)*100}%,
                            #e0e0e0 {(before_ratio+focus_ratio)*100}%,
                            #e0e0e0 100%);
                            height: 40px;
                            border-radius: 10px;
                            border: 2px solid #333;
                            position: relative;
                            margin: 10px 0;">
                            <div style="position: absolute; left: {before_ratio*100}%; top: -25px; font-size: 12px; font-weight: bold;">
                                ⬇️ {rec_start:,}
                            </div>
                            <div style="position: absolute; left: {(before_ratio+focus_ratio)*100}%; top: -25px; font-size: 12px; font-weight: bold;">
                                {rec_end:,} ⬇️
                            </div>
                            <div style="position: absolute; left: 50%; top: 10px; transform: translateX(-50%); color: white; font-weight: bold; font-size: 14px;">
                                🔴 Focus: {rec_range['count']:,} lines
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px; color: #666;">
                            <span>Start: {total_start:,}</span>
                            <span>End: {total_end:,}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # 개별 범위 정보 (여러 범위가 있는 경우)
                        old_format_ranges = experiment_output.get('ranges', [])
                        if len(old_format_ranges) > 1:
                            with st.expander(f"📋 Detailed Ranges ({len(old_format_ranges)} segments)"):
                                for i, (start, end) in enumerate(old_format_ranges, 1):
                                    st.text(f"  Range {i}: lines {start:,} ~ {end:,} ({end-start+1:,} lines)")

                    st.divider()

                # 최종 답변 (하이라이팅 적용)
                st.subheader("📋 최종 답변")

                # [NEW] 로그 라인 범위 하이라이팅 함수
                def highlight_log_ranges(text):
                    """로그 라인 범위를 볼드체와 빨간색으로 강조"""
                    import re

                    # 패턴들
                    patterns = [
                        (r'(line\[\d+~\d+\])', r'**🔴 \1**'),  # line[100~200]
                        (r'(lines?\s+\d+\s*[-~]\s*\d+)', r'**🔴 \1**'),  # line 100-200
                        (r'(\[\d+\s*[-~]\s*\d+\])', r'**🔴 \1**'),  # [100-200]
                        (r'(\d+\s*[-~]\s*\d+\s+lines?)', r'**🔴 \1**'),  # 100-200 lines
                    ]

                    highlighted = text
                    for pattern, replacement in patterns:
                        highlighted = re.sub(pattern, replacement, highlighted, flags=re.IGNORECASE)

                    return highlighted

                # 하이라이팅 적용된 답변 표시
                highlighted_answer = highlight_log_ranges(last_message.content)
                st.markdown(highlighted_answer)

                # 📥 [NEW] CSV 다운로드 기능 (추천 라인 표시)
                st.divider()

                # 디버깅: 상태 확인
                csv_path_exists = st.session_state.uploaded_csv_path is not None
                exp_output_exists = experiment_output is not None

                # [DEBUG] 상태 표시 (개발용)
                with st.expander("🔍 [DEBUG] CSV 다운로드 상태 확인"):
                    st.write(f"CSV 경로 존재: {csv_path_exists}")
                    st.write(f"experiment_output 존재: {exp_output_exists}")
                    if exp_output_exists:
                        st.write(f"experiment_output keys: {list(experiment_output.keys())}")
                        st.write(f"recommended_lines 개수: {len(experiment_output.get('recommended_lines', []))}")

                if csv_path_exists and exp_output_exists:
                    st.subheader("📥 결과 CSV 다운로드")

                    try:
                        import pandas as pd
                        from io import BytesIO

                        # 원본 CSV 로드
                        original_df = pd.read_csv(st.session_state.uploaded_csv_path)

                        # LineId 컬럼이 없으면 생성
                        if 'LineId' not in original_df.columns:
                            original_df.insert(0, 'LineId', range(1, len(original_df) + 1))

                        # 추천 라인 정보 추출
                        rec_range = experiment_output.get("recommended_range")
                        recommended_lines_set = set(experiment_output.get("recommended_lines", []))

                        # 새 컬럼 추가: is_recommended (추천 라인 여부)
                        if rec_range:
                            rec_start = rec_range['start']
                            rec_end = rec_range['end']

                            # 방법 1: 범위 기반
                            original_df['⭐_recommended'] = original_df['LineId'].apply(
                                lambda x: '✓' if rec_start <= x <= rec_end else ''
                            )

                            # 방법 2: 개별 라인 기반 (더 정확)
                            if recommended_lines_set:
                                original_df['⭐_recommended'] = original_df['LineId'].apply(
                                    lambda x: '✓' if x in recommended_lines_set else ''
                                )
                        else:
                            original_df['⭐_recommended'] = ''

                        # CSV를 메모리에 저장
                        csv_buffer = BytesIO()
                        original_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_bytes = csv_buffer.getvalue()

                        # 통계 표시
                        col_dl1, col_dl2, col_dl3 = st.columns([2, 2, 1])

                        with col_dl1:
                            st.metric("전체 로그 라인 수", f"{len(original_df):,}")

                        with col_dl2:
                            recommended_count = (original_df['⭐_recommended'] == '✓').sum()
                            st.metric("추천 라인 수", f"{recommended_count:,}",
                                     delta=f"{recommended_count/len(original_df)*100:.1f}%")

                        with col_dl3:
                            # 다운로드 버튼
                            st.download_button(
                                label="📥 CSV 다운로드",
                                data=csv_bytes,
                                file_name=f"analysis_result_{Path(st.session_state.uploaded_csv_path).stem}.csv",
                                mime="text/csv",
                                help="추천 라인이 '⭐_recommended' 컬럼에 ✓ 표시되어 있습니다."
                            )

                        # 전체 로그 미리보기 (권장 범위 하이라이트)
                        st.subheader("📄 전체 로그 라인 (권장 범위 하이라이트)")
                        st.caption(f"총 {len(original_df):,}행 전체 표시")

                        if '⭐_recommended' in original_df.columns:
                            def highlight_rows(row):
                                color = '#ffe5e5' if row['⭐_recommended'] == '✓' else ''
                                return ['background-color: {}'.format(color) if color else '' for _ in row]

                            styled_preview = original_df.style.apply(highlight_rows, axis=1)
                            st.dataframe(styled_preview, use_container_width=True, height=600)
                            st.caption("🔴 붉게 표시된 행이 권장 로그 범위에 속한 라인입니다.")
                        else:
                            st.dataframe(original_df, use_container_width=True, height=600)
                            st.caption("권장 범위 정보가 없어 일반 테이블로 표시합니다.")

                    except Exception as csv_error:
                        st.error(f"CSV 생성 실패: {csv_error}")
                        with st.expander("CSV 오류 상세"):
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    # 조건을 만족하지 않을 때
                    if not csv_path_exists:
                        st.info("ℹ️ CSV 파일을 업로드하지 않았습니다. CSV 다운로드를 사용하려면 로그 파일을 업로드하세요.")
                    elif not exp_output_exists:
                        st.warning("⚠️ experiment_output이 생성되지 않았습니다. answer.py가 최신 버전인지 확인하세요.")

                # 시각화 자료 (CSV가 업로드된 경우에만)
                if st.session_state.uploaded_csv_path:
                    st.divider()
                    with st.expander("📈 시각화 자료 보기", expanded=False):
                        try:
                            with st.spinner("📊 시각화 생성 중... 차트를 그리고 있어요!"):
                                viz_path = create_visualization(st.session_state.uploaded_csv_path)

                                if Path(viz_path).exists():
                                    st.image(viz_path, use_container_width=True)
                                    st.caption("로그 데이터 종합 분석 대시보드")
                                else:
                                    st.warning("시각화 이미지를 생성하지 못했습니다.")

                        except Exception as viz_error:
                            st.error(f"시각화 생성 실패: {viz_error}")
                            with st.expander("시각화 오류 상세"):
                                import traceback
                                st.code(traceback.format_exc())

                # 메타데이터
                with st.expander("📊 분석 메타데이터"):
                    metadata = final_state.get("metadata", {})

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("FAISS 호출", metadata.get("faiss_calls", 0))
                    with col_m2:
                        st.metric("Planner 반복", metadata.get("planner_iterations", 0))
                    with col_m3:
                        st.metric("총 메시지", len(final_state.get("messages", [])))

                    st.json(metadata)

                # 전체 대화 히스토리
                with st.expander("💬 전체 대화 히스토리"):
                    for i, msg in enumerate(final_state.get("messages", []), 1):
                        ak = getattr(msg, "additional_kwargs", {}) or getattr(msg, "kwargs", {}) or {}
                        agent_name = ak.get("agent_name", "unknown")

                        st.text(f"[{i}] {msg.__class__.__name__} from {agent_name}")
                        st.code(msg.content, language="text")

        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            with st.expander("상세 오류 정보"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("LORIN Log Analyzer v1.0 - 로컬호스트 버전")
