# FAISS 인덱스 구축 (BAAI/bge-multilingual-gemma2 - Multi-GPU 지원)
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import math
import gc

from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# ── 중앙 집중식 설정 로드 ───────────────────────────────────
from LORIN.config.settings import get_faiss_settings, get_retrieval_settings

# ── 설정 기반 파라미터 (환경변수로 오버라이드 가능) ─────────
_faiss_cfg = get_faiss_settings()
_retrieval_cfg = get_retrieval_settings()

WIN = _faiss_cfg.window_size
STRIDE = _faiss_cfg.stride
MIN_LEN = _faiss_cfg.min_len
INDEX_PATH = _retrieval_cfg.index_path
MODEL_NAME = _retrieval_cfg.model_name
MAX_TOKEN_LENGTH = _faiss_cfg.max_token_length

# GPU 설정
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    print(f"Found {GPU_COUNT} GPU(s)")
    for i in range(GPU_COUNT):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    DEVICE = torch.device("cuda")
else:
    GPU_COUNT = 0
    DEVICE = torch.device("cpu")
    print("No GPU found, using CPU")

# ── Custom BGE-Gemma2 Multi-GPU Embeddings Class ───────────
class BGEGemma2MultiGPUEmbeddings(Embeddings):
    """BAAI/bge-multilingual-gemma2 임베딩을 위한 멀티 GPU 지원 클래스"""
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        use_multi_gpu: bool = True,
        gpu_ids: Optional[List[int]] = None,
        batch_size_per_gpu: int = 16,
        use_fp16: bool = True
    ):
        """
        Args:
            model_name: HuggingFace 모델 이름
            use_multi_gpu: 멀티 GPU 사용 여부
            gpu_ids: 사용할 GPU ID 리스트 (None이면 모든 GPU 사용)
            batch_size_per_gpu: GPU당 배치 크기
            use_fp16: FP16 (half precision) 사용 여부 - 메모리 절약 및 속도 향상
        """
        self.model_name = model_name
        self.use_multi_gpu = use_multi_gpu and GPU_COUNT > 1
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        # GPU IDs 설정
        if gpu_ids is None and torch.cuda.is_available():
            self.gpu_ids = list(range(GPU_COUNT))
        else:
            self.gpu_ids = gpu_ids if gpu_ids else [0]
        
        # 배치 크기 설정
        self.batch_size_per_gpu = batch_size_per_gpu
        self.total_batch_size = batch_size_per_gpu * len(self.gpu_ids) if self.use_multi_gpu else batch_size_per_gpu
        
        print(f"Using GPUs: {self.gpu_ids}")
        print(f"Total batch size: {self.total_batch_size}")
        
        # 모델과 토크나이저 로드
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32
        )
        
        # Multi-GPU 설정
        if self.use_multi_gpu:
            print(f"Setting up DataParallel on GPUs: {self.gpu_ids}")
            # 첫 번째 GPU를 primary로 설정
            torch.cuda.set_device(self.gpu_ids[0])
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            self.model.to(f'cuda:{self.gpu_ids[0]}')
        else:
            device = f'cuda:{self.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
        
        self.model.eval()
        
        # 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # BGE 모델의 경우 쿼리와 문서에 다른 prefix 사용
        self.query_instruction = "Represent this sentence for searching relevant passages: "
    
    def _get_base_model(self):
        """DataParallel로 감싸진 경우 기본 모델 반환"""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model
    
    def _mean_pooling(self, model_output, attention_mask):
        """평균 풀링을 통한 sentence embedding 생성"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _encode_batch(self, texts: List[str], add_instruction: bool = False, log_stats: bool = False) -> np.ndarray:
        """단일 배치 인코딩 (GPU 메모리 효율적 처리)"""
        # Instruction 추가
        if add_instruction:
            texts = [self.query_instruction + text for text in texts]

        # 토크나이징
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt'
        )

        # 통계 로깅 (첫 번째 배치에서만)
        if log_stats and len(texts) > 0:
            input_ids = encoded_input['input_ids']
            actual_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
            max_actual = actual_lengths.max().item()
            avg_actual = actual_lengths.float().mean().item()
            truncated = (actual_lengths >= MAX_TOKEN_LENGTH).sum().item()

            print(f"\n=== Token Statistics (First Batch) ===")
            print(f"  Batch size: {len(texts)}")
            print(f"  Max token length: {MAX_TOKEN_LENGTH}")
            print(f"  Max tokens in batch: {max_actual}")
            print(f"  Avg tokens in batch: {avg_actual:.1f}")
            print(f"  Truncated documents: {truncated}/{len(texts)}")
            if truncated > 0:
                print(f"  ⚠️  Warning: {truncated} documents exceeded {MAX_TOKEN_LENGTH} token limit and were truncated")

        # GPU로 이동
        if torch.cuda.is_available():
            device = f'cuda:{self.gpu_ids[0]}'
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # 모델 추론
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                model_output = self.model(**encoded_input)
        
        # 평균 풀링
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # L2 정규화
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _encode(self, texts: List[str], add_instruction: bool = False, show_progress: bool = True) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환 (멀티 GPU 최적화)"""
        if not texts:
            return np.array([])

        all_embeddings = []

        # 프로그레스 바 설정
        disable_progress = not show_progress or len(texts) < 100

        # 배치 처리
        for i in tqdm(range(0, len(texts), self.total_batch_size),
                     desc="Encoding", disable=disable_progress):
            batch_texts = texts[i:i + self.total_batch_size]

            try:
                # 배치 인코딩 (첫 번째 배치에서만 통계 로깅)
                log_stats = (i == 0 and not add_instruction)  # 문서 임베딩 시에만, 첫 배치에만
                embeddings = self._encode_batch(batch_texts, add_instruction, log_stats=log_stats)
                all_embeddings.append(embeddings)
                
                # 각 배치마다 완전한 GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # GPU 작업 완료 대기
                    torch.cuda.empty_cache()  # GPU 캐시 비우기
                    gc.collect()              # Python GC 강제 실행
                    
            except torch.cuda.OutOfMemoryError:
                print(f"GPU OOM at batch {i}. Trying smaller batch size...")
                # 완전한 메모리 정리 후 재시도
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                
                # 더 작은 배치로 재시도
                small_batch_size = self.batch_size_per_gpu // 2
                for j in range(0, len(batch_texts), small_batch_size):
                    small_batch = batch_texts[j:j + small_batch_size]
                    embeddings = self._encode_batch(small_batch, add_instruction)
                    all_embeddings.append(embeddings)
        
        # 최종 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 (instruction 없이)"""
        embeddings = self._encode(texts, add_instruction=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 (instruction 포함)"""
        embeddings = self._encode([text], add_instruction=True, show_progress=False)
        return embeddings[0].tolist()

# ── 고급 멀티 GPU 병렬 처리 클래스 (선택사항) ─────────────
class BGEGemma2AdvancedMultiGPU(BGEGemma2MultiGPUEmbeddings):
    """더 효율적인 멀티 GPU 처리를 위한 고급 클래스"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = []
        
        if self.use_multi_gpu and GPU_COUNT > 1:
            print("Setting up independent models on each GPU...")
            # 각 GPU에 독립적인 모델 인스턴스 생성
            for gpu_id in self.gpu_ids:
                model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32
                )
                model.to(f'cuda:{gpu_id}')
                model.eval()
                self.models.append(model)
            
            # 기존 DataParallel 모델은 사용하지 않음
            del self.model
            torch.cuda.empty_cache()
    
    def _encode_on_gpu(self, texts: List[str], gpu_id: int, model, add_instruction: bool, log_stats: bool = False) -> np.ndarray:
        """특정 GPU에서 인코딩 수행"""
        if add_instruction:
            texts = [self.query_instruction + text for text in texts]

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt'
        ).to(f'cuda:{gpu_id}')

        # 통계 로깅 (첫 번째 GPU의 첫 번째 배치에서만)
        if log_stats and gpu_id == self.gpu_ids[0] and len(texts) > 0:
            input_ids = encoded_input['input_ids']
            actual_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
            max_actual = actual_lengths.max().item()
            avg_actual = actual_lengths.float().mean().item()
            truncated = (actual_lengths >= MAX_TOKEN_LENGTH).sum().item()

            print(f"\n=== Token Statistics (First Batch, GPU {gpu_id}) ===")
            print(f"  Batch size: {len(texts)}")
            print(f"  Max token length: {MAX_TOKEN_LENGTH}")
            print(f"  Max tokens in batch: {max_actual}")
            print(f"  Avg tokens in batch: {avg_actual:.1f}")
            print(f"  Truncated documents: {truncated}/{len(texts)}")
            if truncated > 0:
                print(f"  ⚠️  Warning: {truncated} documents exceeded {MAX_TOKEN_LENGTH} token limit and were truncated")

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                model_output = model(**encoded_input)

        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
    
    def _encode(self, texts: List[str], add_instruction: bool = False, show_progress: bool = True) -> np.ndarray:
        """멀티 GPU 병렬 인코딩"""
        if not self.models:
            # 기본 구현 사용
            return super()._encode(texts, add_instruction, show_progress)
        
        all_embeddings = []
        disable_progress = not show_progress or len(texts) < 100
        
        # GPU별로 텍스트 분할
        texts_per_gpu = math.ceil(len(texts) / len(self.gpu_ids))
        
        with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            futures = []

            for gpu_idx, gpu_id in enumerate(self.gpu_ids):
                start_idx = gpu_idx * texts_per_gpu
                end_idx = min(start_idx + texts_per_gpu, len(texts))

                if start_idx < len(texts):
                    gpu_texts = texts[start_idx:end_idx]
                    # 첫 번째 GPU에서만 통계 로깅 (문서 임베딩 시에만)
                    log_stats = (gpu_idx == 0 and not add_instruction)
                    future = executor.submit(
                        self._encode_on_gpu,
                        gpu_texts, gpu_id, self.models[gpu_idx], add_instruction, log_stats
                    )
                    futures.append(future)

            # 결과 수집
            for future in tqdm(futures, desc="GPU Processing", disable=disable_progress):
                embeddings = future.result()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

# ── NaN 안전 처리 함수들 ────────────────────────────────────
def safe_str(v):
    return "" if pd.isna(v) else str(v)

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

# ── 연속 LineId 그룹 전용 슬라이딩 윈도우 (기존 LineId 절대 보존) ──────────
def sliding_chunks_consecutive_only(records, win=WIN, stride=STRIDE, min_len=MIN_LEN):
    """
    연속된 LineId 그룹 내에서만 청크 생성 - 기존 LineId 절대 변경 금지

    핵심 원칙:
    1. 기존 LineId 절대 수정 금지
    2. 연속된 LineId 그룹만 청크로 구성
    3. 비연속 그룹 간 청크 생성 금지

    Args:
        records: DataFrame records (dict 형태) - 원본 LineId 보존
        win: 윈도우 크기 (레코드 개수)
        stride: 스트라이드 (레코드 개수)
        min_len: 최소 청크 크기

    Yields:
        연속된 LineId 청크들 (기존 LineId 그대로 유지)
    """
    if not records:
        print("❌ 레코드 없음")
        return

    # 1. LineId 기준 정렬 (원본 데이터 보존)
    sorted_records = sorted(records, key=lambda x: safe_int(x.get("LineId", 0)))
    available_lineids = [safe_int(r.get("LineId", 0)) for r in sorted_records]

    print(f"\n=== 연속 LineId 그룹 분석 ===")
    print(f"LineId 범위: {min(available_lineids)} ~ {max(available_lineids)}")
    print(f"총 레코드 수: {len(available_lineids)}개")

    # 2. 연속된 LineId 그룹 식별
    consecutive_groups = []
    current_group = [sorted_records[0]]
    gaps_found = []

    for i in range(1, len(sorted_records)):
        prev_lineid = safe_int(sorted_records[i-1].get("LineId", 0))
        curr_lineid = safe_int(sorted_records[i].get("LineId", 0))
        gap = curr_lineid - prev_lineid

        if gap == 1:  # 연속
            current_group.append(sorted_records[i])
        else:  # 비연속
            gaps_found.append((prev_lineid, curr_lineid, gap))

            # 이전 그룹이 최소 크기 충족시 저장
            if len(current_group) >= min_len:
                consecutive_groups.append({
                    'records': current_group.copy(),
                    'start_lineid': safe_int(current_group[0].get("LineId", 0)),
                    'end_lineid': safe_int(current_group[-1].get("LineId", 0)),
                    'size': len(current_group)
                })

            # 새 그룹 시작
            current_group = [sorted_records[i]]

    # 마지막 그룹 처리
    if len(current_group) >= min_len:
        consecutive_groups.append({
            'records': current_group.copy(),
            'start_lineid': safe_int(current_group[0].get("LineId", 0)),
            'end_lineid': safe_int(current_group[-1].get("LineId", 0)),
            'size': len(current_group)
        })

    print(f"\n=== 연속성 분석 결과 ===")
    print(f"비연속 구간 수: {len(gaps_found)}개")
    print(f"연속 그룹 수: {len(consecutive_groups)}개")

    if gaps_found:
        print(f"주요 비연속 구간: {gaps_found[:5]}")

    # 3. 각 연속 그룹에서 청크 생성
    total_chunks = 0

    for group_idx, group_info in enumerate(consecutive_groups):
        group_records = group_info['records']
        start_id = group_info['start_lineid']
        end_id = group_info['end_lineid']
        group_size = group_info['size']

        print(f"\n그룹 {group_idx + 1}: LineId {start_id}~{end_id} ({group_size}개 레코드)")

        # 연속성 재검증
        group_lineids = [safe_int(r.get("LineId", 0)) for r in group_records]
        if not _is_perfectly_consecutive(group_lineids):
            print(f"❌ 그룹 내 비연속성 발견, 스킵")
            continue

        # 그룹 내 슬라이딩 윈도우 (연속된 범위만)
        group_chunks = 0
        for s in range(0, group_size, stride):
            chunk_end = min(s + win, group_size)
            chunk = group_records[s:chunk_end]

            if len(chunk) >= min_len:
                # 청크 내 연속성 최종 검증
                chunk_lineids = [safe_int(r.get("LineId", 0)) for r in chunk]

                if _is_perfectly_consecutive(chunk_lineids):
                    total_chunks += 1
                    group_chunks += 1
                    yield chunk
                else:
                    print(f"❌ 청크 내 비연속성: {chunk_lineids[:3]}...{chunk_lineids[-3:]}")

        print(f"  → {group_chunks}개 청크 생성")

    print(f"\n=== 최종 결과 ===")
    print(f"총 {total_chunks}개 연속 LineId 청크 생성")
    print(f"기존 LineId 100% 보존 보장")

def _is_perfectly_consecutive(lineids):
    """완벽한 연속성 검증 - 1씩 증가하는지 확인"""
    if len(lineids) <= 1:
        return True

    for i in range(1, len(lineids)):
        if lineids[i] != lineids[i-1] + 1:
            return False
    return True

# ── 기존 호환성을 위한 래퍼 함수 (연속성 전용) ─────────────────────────
def sliding_chunks(lines, win=WIN, stride=STRIDE, min_len=MIN_LEN):
    """
    기존 호환성 래퍼 - 연속 LineId 그룹만 청크로 구성
    기존 LineId 절대 변경 금지
    """
    return sliding_chunks_consecutive_only(lines, win, stride, min_len)

# ── Content 전용 page_content ─────────────────────────────
def to_content(row: dict) -> str:
    comp = safe_str(row.get("Component", "")).strip()
    text = safe_str(row.get("Content", ""))
    return f"[{comp}] {text}" if comp else text


# ── 연속 LineId 전용 문서 생성 (기존 LineId 절대 보존) ──────────────────
def build_docs_from_df(df: pd.DataFrame, total_lines: Optional[int] = None):
    """
    연속된 LineId 그룹만 청크로 구성 - 기존 LineId 절대 변경 금지

    핵심 원칙:
    1. 기존 LineId 그대로 유지 (Range = [실제_첫번째_LineId, 실제_마지막_LineId])
    2. 비연속 구간은 청크로 만들지 않음
    3. 연속성 100% 보장
    4. Temporal filtering을 위한 relative_position 메타데이터 추가

    Args:
        df: 입력 DataFrame
        total_lines: CSV 전체 라인 수 (None이면 DataFrame에서 최대 LineId로 추정)
    """
    records = df.to_dict(orient="records")
    docs = []
    chunk_count = 0

    # CSV 전체 라인 수 계산
    if total_lines is None:
        # DataFrame에서 최대 LineId 추출 (실제 로그 파일의 라인 수)
        all_lineids = [safe_int(r.get("LineId", 0)) for r in records]
        total_lines = max(all_lineids) if all_lineids else len(records)

    print(f"\n=== 연속 LineId 청크 생성 ===")
    print(f"총 레코드 수: {len(records)}")
    print(f"CSV 전체 라인 수 (로그): {total_lines}")
    print(f"기존 LineId 절대 보존 모드")
    print(f"✨ Temporal metadata 추가: relative_position, line_number")

    for ch_idx, ch in enumerate(sliding_chunks_consecutive_only(records)):

        # 청크의 첫번째와 마지막 레코드에서 LineId 추출
        md0, mdN = ch[0], ch[-1]
        line_start = safe_int(md0.get("LineId"))  # 실제 LineId 그대로 사용
        line_end = safe_int(mdN.get("LineId"))    # 실제 LineId 그대로 사용
        time_start = safe_str(md0.get("Time"))
        time_end = safe_str(mdN.get("Time"))

        # 청크 내 모든 LineId 추출 (연속성 보장됨)
        chunk_lineids = [safe_int(r.get("LineId", 0)) for r in ch]

        # 연속성 최종 검증 (이미 알고리즘에서 보장되지만 이중 체크)
        if not _is_perfectly_consecutive(chunk_lineids):
            print(f"❌ 비연속 청크 발견 (LineId: {chunk_lineids[:3]}...{chunk_lineids[-3:]}), 스킵")
            continue

        # 예상된 Range와 실제 LineId 범위 비교
        expected_lineids = list(range(line_start, line_end + 1))
        if chunk_lineids != expected_lineids:
            print(f"❌ LineId 불일치: 예상 {expected_lineids[:3]}...{expected_lineids[-3:]}, 실제 {chunk_lineids[:3]}...{chunk_lineids[-3:]}")
            continue

        # page_content 생성
        content_text = "\n".join(to_content(r) for r in ch)

        # ✨ Temporal filtering을 위한 메타데이터 계산
        line_number = (line_start + line_end) // 2  # 청크 중간 라인 번호
        relative_position = line_number / total_lines if total_lines > 0 else 0.0

        # 메타데이터 구성
        metadata = {
            "TimeStart": time_start,
            "TimeEnd": time_end,
            "Range": [line_start, line_end],  # 기존 LineId 그대로
            "ActualLineIds": chunk_lineids,      # 실제 LineId 목록
            "ChunkSize": len(ch),
            "ConsecutiveGuaranteed": True,        # 연속성 100% 보장
            "OriginalLineIdsPreserved": True,     # 기존 LineId 보존 보장
            # ✨ Temporal filtering 메타데이터 (NEW)
            "line_number": line_number,           # 청크 중간 라인 번호
            "total_lines": total_lines,           # CSV 전체 라인 수
            "relative_position": relative_position  # 0.0 ~ 1.0 (temporal 위치)
        }

        # 첫 번째 청크의 메타데이터 상세 로깅
        if chunk_count == 0:
            print(f"\n=== 첫 번째 청크 메타데이터 샘플 ===")
            print(f"Temporal Metadata Fields:")
            print(f"  - line_number: {metadata['line_number']} (청크 중간 라인)")
            print(f"  - total_lines: {metadata['total_lines']} (전체 로그 라인 수)")
            print(f"  - relative_position: {metadata['relative_position']:.4f} (0.0~1.0)")
            print(f"\nLegacy Metadata Fields:")
            print(f"  - TimeStart: {metadata['TimeStart']}")
            print(f"  - TimeEnd: {metadata['TimeEnd']}")
            print(f"  - Range: {metadata['Range']}")
            print(f"  - ChunkSize: {metadata['ChunkSize']}")
            print(f"\nAll Metadata Keys: {list(metadata.keys())}")
            print(f"{'='*60}\n")

        # 문서 생성 (기존 LineId 그대로 유지 + temporal metadata 추가)
        docs.append(
            Document(
                page_content=content_text,
                metadata=metadata
            )
        )
        chunk_count += 1

        # 주기적 진행 상황 로깅 (매 100개 청크마다)
        if chunk_count % 100 == 0:
            print(f"Progress: {chunk_count} chunks created | "
                  f"Latest chunk: line_number={line_number}, "
                  f"relative_position={relative_position:.4f}")

    print(f"\n=== 청크 생성 결과 ===")
    print(f"생성된 문서 수: {chunk_count}")
    print(f"LineId 연속성: 100% 보장")
    print(f"기존 LineId 보존: 100% 보장")
    print(f"비연속 구간 청크 생성: 0개 (전적 차단)")

    # 마지막 청크의 메타데이터 출력 (temporal range 확인용)
    if docs:
        last_doc = docs[-1]
        last_metadata = last_doc.metadata
        print(f"\n=== 마지막 청크 메타데이터 ===")
        print(f"  - line_number: {last_metadata.get('line_number', 'N/A')}")
        print(f"  - relative_position: {last_metadata.get('relative_position', 0.0):.4f}")
        print(f"  - Range: {last_metadata.get('Range', 'N/A')}")
        print(f"  - total_lines: {last_metadata.get('total_lines', 'N/A')}")

        # Temporal distribution 요약
        first_doc = docs[0]
        first_rel_pos = first_doc.metadata.get('relative_position', 0.0)
        last_rel_pos = last_metadata.get('relative_position', 0.0)
        print(f"\n=== Temporal Distribution ===")
        print(f"  - First chunk position: {first_rel_pos:.4f} ({first_rel_pos*100:.1f}%)")
        print(f"  - Last chunk position: {last_rel_pos:.4f} ({last_rel_pos*100:.1f}%)")
        print(f"  - Coverage: {first_rel_pos:.4f} ~ {last_rel_pos:.4f}")

    return docs


# ── 빌드 & 저장 (멀티 GPU 지원) ───────────────────────────
def build_and_save_faiss(
    df: pd.DataFrame,
    index_path: str = INDEX_PATH,
    use_advanced_multi_gpu: bool = False,
    batch_size_per_gpu: int = 32
):
    """
    DataFrame로부터 FAISS 인덱스 생성 및 저장

    Args:
        df: 입력 DataFrame
        index_path: 인덱스 저장 경로
        use_advanced_multi_gpu: 고급 멀티 GPU 병렬 처리 사용 여부
        batch_size_per_gpu: GPU당 배치 크기
    """

    # 설정값 출력
    print("\n=== FAISS Index Build Configuration ===")
    print(f"Chunking parameters:")
    print(f"  WIN (window size): {WIN} lines")
    print(f"  STRIDE (overlap): {STRIDE} lines")
    print(f"  MIN_LEN (minimum): {MIN_LEN} lines")
    print(f"Embedding parameters:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max token length: {MAX_TOKEN_LENGTH} tokens")
    print(f"  Batch size per GPU: {batch_size_per_gpu}")
    print(f"  Advanced multi-GPU: {use_advanced_multi_gpu}")

    # GPU 정보 출력
    if torch.cuda.is_available():
        print("\n=== GPU Information ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Current Usage: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
    
    # 임베딩 모델 선택
    if use_advanced_multi_gpu and GPU_COUNT > 1:
        print("\n사용: Advanced Multi-GPU Processing")
        embeddings = BGEGemma2AdvancedMultiGPU(
            model_name=MODEL_NAME,
            batch_size_per_gpu=batch_size_per_gpu,
            use_fp16=True
        )
    else:
        print("\n사용: Standard Multi-GPU (DataParallel)")
        embeddings = BGEGemma2MultiGPUEmbeddings(
            model_name=MODEL_NAME,
            batch_size_per_gpu=batch_size_per_gpu,
            use_fp16=True
        )
    
    # CSV 전체 라인 수 계산 (temporal filtering을 위해 필요)
    all_lineids = df["LineId"].dropna().astype(int)
    total_lines = int(all_lineids.max()) if len(all_lineids) > 0 else len(df)
    print(f"CSV 전체 라인 수: {total_lines:,}")

    # 문서 생성 (total_lines 전달)
    docs = build_docs_from_df(df, total_lines=total_lines)
    print(f"생성된 청크 수: {len(docs):,}")
    
    # FAISS 인덱스 생성
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 저장
    vectorstore.save_local(index_path)
    print(f"\n=== FAISS 인덱스 저장 완료 ===")
    print(f"저장 경로: {index_path}")

    # ✨ 저장된 인덱스 검증: 메타데이터 필드 확인
    print(f"\n=== 메타데이터 검증 (저장 직후) ===")
    try:
        # 샘플 문서 추출 (첫 번째 문서)
        sample_docs = vectorstore.similarity_search("test", k=1)
        if sample_docs:
            sample_metadata = sample_docs[0].metadata
            print(f"✅ 메타데이터 필드 확인:")
            print(f"  - line_number: {'✓' if 'line_number' in sample_metadata else '✗ MISSING'}")
            print(f"  - total_lines: {'✓' if 'total_lines' in sample_metadata else '✗ MISSING'}")
            print(f"  - relative_position: {'✓' if 'relative_position' in sample_metadata else '✗ MISSING'}")
            print(f"  - Range: {'✓' if 'Range' in sample_metadata else '✗ MISSING'}")
            print(f"  - TimeStart: {'✓' if 'TimeStart' in sample_metadata else '✗ MISSING'}")

            # 실제 값 출력
            if 'relative_position' in sample_metadata:
                print(f"\n샘플 값:")
                print(f"  - line_number: {sample_metadata.get('line_number', 'N/A')}")
                print(f"  - total_lines: {sample_metadata.get('total_lines', 'N/A')}")
                print(f"  - relative_position: {sample_metadata.get('relative_position', 'N/A'):.4f}")
                print(f"  - Range: {sample_metadata.get('Range', 'N/A')}")
            else:
                print(f"\n⚠️ WARNING: temporal metadata 필드가 없습니다!")
                print(f"Available fields: {list(sample_metadata.keys())}")
        else:
            print(f"⚠️ WARNING: 샘플 문서를 찾을 수 없습니다.")
    except Exception as e:
        print(f"⚠️ 메타데이터 검증 중 오류: {e}")

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vectorstore

# ── 검색 함수 (멀티 GPU 지원) ────────────────────────────
def load_and_search(
    query: str, 
    index_path: str = INDEX_PATH, 
    k: int = 15,
    use_advanced_multi_gpu: bool = False
):
    """저장된 FAISS 인덱스를 로드하고 검색 수행"""
    
    # 임베딩 초기화
    print("Loading embeddings...")
    if use_advanced_multi_gpu and GPU_COUNT > 1:
        embeddings = BGEGemma2AdvancedMultiGPU(
            model_name=MODEL_NAME,
            batch_size_per_gpu=32,
            use_fp16=True
        )
    else:
        embeddings = BGEGemma2MultiGPUEmbeddings(
            model_name=MODEL_NAME,
            batch_size_per_gpu=32,
            use_fp16=True
        )
    
    # FAISS 인덱스 로드
    print(f"Loading FAISS index from {index_path}...")
    vectorstore = FAISS.load_local(
        index_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 검색 수행
    print(f"Searching for: {query}")
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    return results

# ── GPU 메모리 모니터링 유틸리티 ─────────────────────────
def print_gpu_memory_usage():
    """현재 GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        print("\n=== GPU Memory Usage ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Total: {total:.2f} GB")
            print(f"  Free: {total - allocated:.2f} GB")
