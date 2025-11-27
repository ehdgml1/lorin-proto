#!/usr/bin/env python3
"""
FAISS 검색 엔진 - BGE-multilingual-gemma2 기반
쿼리를 입력받아 관련 문서를 검색하고 결과를 출력하는 시스템
"""

import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from colorama import init, Fore, Back, Style
import argparse

# Colorama 초기화 (Windows 지원)
init(autoreset=True)

# ── 환경 설정 ─────────────────────────────────────────────
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Tokenizer parallelism 경고 방지
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── 검색 결과 데이터 클래스 ────────────────────────────────
@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""
    rank: int
    score: float
    content: str
    metadata: Dict[str, Any]
    line_range: List[int]  # [line_start, line_end]
    time_start: str
    time_end: str

# ── FAISS 검색 엔진 클래스 ─────────────────────────────────
class FAISSSearchEngine:
    """FAISS 기반 검색 엔진"""
    
    def __init__(
        self,
        index_path: str,
        model_name: str = "BAAI/bge-multilingual-gemma2",
        device: str = None,
        use_fp16: bool = True,
        max_length: int = 2048,
        quantization: str = "none"  # "4bit", "8bit", "none"
    ):
        """
        Args:
            index_path: FAISS 인덱스 경로
            model_name: 임베딩 모델 이름
            device: 사용할 디바이스 (None이면 자동 선택)
            use_fp16: FP16 사용 여부 (quantization="none"일 때만 사용)
            max_length: 토크나이저 최대 길이
            quantization: 양자화 모드 ("4bit": 최대 절약/느림, "8bit": 균형/빠름, "none": 양자화 없음)
        """
        self.index_path = index_path
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.quantization = quantization

        # 디바이스 설정 (BGE 임베딩을 GPU 0로 - EXAONE은 Multi-GPU 분산)
        if device is None:
            # GPU 0 사용 (EXAONE은 GPU 0+1에 분산)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"{Fore.CYAN}디바이스: {self.device} (BGE 임베딩 모델)")
        
        # 모델과 인덱스 로드
        self._load_model()
        self._load_index()
    
    def _load_model(self):
        """임베딩 모델 로드 (선택적 양자화 지원)"""
        from transformers import AutoTokenizer, AutoModel

        print(f"{Fore.YELLOW}모델 로딩 중: {self.model_name}...")

        # 로딩 전 GPU 메모리 상태
        if torch.cuda.is_available():
            device_idx = int(str(self.device).split(':')[-1]) if ':' in str(self.device) else 0
            initial_allocated = torch.cuda.memory_allocated(device_idx) / 1e9
            print(f"{Fore.CYAN}로딩 전 GPU {device_idx} 메모리: {initial_allocated:.2f}GB")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 양자화 모드에 따른 설정
        if self.quantization == "4bit":
            # 🔧 4-bit 양자화: 최대 메모리 절약 (9GB → 2-3GB), 로딩 느림
            print(f"{Fore.CYAN}✓ 4-bit 양자화 활성화 (최대 메모리 절약, 로딩 느림)")

            # device_map을 단순 문자열로 지정 (bitsandbytes 호환)
            device_idx = int(str(self.device).split(':')[-1]) if ':' in str(self.device) else 0

            # BitsAndBytesConfig는 load_in_4bit와 충돌할 수 있으므로 직접 플래그 사용
            self.model = AutoModel.from_pretrained(
                self.model_name,
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                device_map={"": device_idx},
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            quant_info = "4-bit quantized"

        elif self.quantization == "8bit":
            # 🔧 8-bit 양자화: 균형잡힌 선택 (9GB → 4-5GB), 로딩 빠름 (추천!)
            print(f"{Fore.CYAN}✓ 8-bit 양자화 활성화 (균형 모드, 로딩 빠름) ⚡")

            # device_map을 단순 문자열로 지정 (bitsandbytes 호환)
            device_idx = int(str(self.device).split(':')[-1]) if ':' in str(self.device) else 0

            self.model = AutoModel.from_pretrained(
                self.model_name,
                load_in_8bit=True,  # 직접 플래그 사용 (BitsAndBytesConfig 없이)
                device_map={"": device_idx},  # GPU 인덱스로 지정
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            quant_info = "8-bit quantized"

        else:
            # 양자화 없음: 원본 모델 (9-10GB), 로딩 가장 빠름
            print(f"{Fore.CYAN}양자화 없음 (원본 모델, FP16: ~9GB)")

            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                trust_remote_code=True
            )
            self.model.to(self.device)
            quant_info = "FP16" if self.use_fp16 else "FP32"

        self.model.eval()

        # BGE 모델용 쿼리 prefix
        self.query_instruction = "Represent this sentence for searching relevant passages: "

        print(f"{Fore.GREEN}✓ 모델 로드 완료 ({quant_info})")

        # 메모리 사용량 출력
        if torch.cuda.is_available():
            device_idx = int(str(self.device).split(':')[-1]) if ':' in str(self.device) else 0
            allocated = torch.cuda.memory_allocated(device_idx) / 1e9
            reserved = torch.cuda.memory_reserved(device_idx) / 1e9
            memory_increase = allocated - initial_allocated
            print(f"{Fore.CYAN}로딩 후 GPU {device_idx} 메모리 - Allocated: {allocated:.2f}GB (+{memory_increase:.2f}GB), Reserved: {reserved:.2f}GB")

            # 양자화 효과 검증
            if self.quantization in ["4bit", "8bit"]:
                expected_mem = 2.5 if self.quantization == "4bit" else 4.5
                if memory_increase > expected_mem + 1.5:  # 1.5GB 마진
                    print(f"{Fore.RED}⚠️  경고: 양자화가 제대로 적용되지 않았을 수 있습니다!")
                    print(f"{Fore.RED}   예상: ~{expected_mem:.1f}GB, 실제: {memory_increase:.2f}GB")
                else:
                    print(f"{Fore.GREEN}✓ 양자화 성공: {memory_increase:.2f}GB (예상 범위)")
    
    def _load_index(self):
        """FAISS 인덱스 로드"""
        from langchain_community.vectorstores import FAISS
        from langchain.embeddings.base import Embeddings
        
        # 커스텀 임베딩 클래스
        class CustomEmbeddings(Embeddings):
            def __init__(self, engine):
                self.engine = engine
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self.engine._encode_text(text, add_instruction=False) for text in texts]
            
            def embed_query(self, text: str) -> List[float]:
                return self.engine._encode_text(text, add_instruction=True)
        
        print(f"{Fore.YELLOW}인덱스 로딩 중: {self.index_path}...")

        embeddings = CustomEmbeddings(self)
        self.vectorstore = FAISS.load_local(
            self.index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"{Fore.GREEN}✓ 인덱스 로드 완료")

        # ✨ 로드된 인덱스의 메타데이터 검증
        print(f"\n{Fore.CYAN}=== 로드된 인덱스 메타데이터 검증 ===")
        try:
            # 샘플 문서 추출 (첫 번째 문서)
            sample_docs = self.vectorstore.similarity_search("test", k=1)
            if sample_docs:
                sample_metadata = sample_docs[0].metadata

                # Temporal metadata 필드 확인
                has_line_number = 'line_number' in sample_metadata
                has_total_lines = 'total_lines' in sample_metadata
                has_relative_position = 'relative_position' in sample_metadata
                has_range = 'Range' in sample_metadata

                print(f"{Fore.CYAN}메타데이터 필드 상태:")
                print(f"  - line_number: {Fore.GREEN + '✓' if has_line_number else Fore.RED + '✗ MISSING'}")
                print(f"  - total_lines: {Fore.GREEN + '✓' if has_total_lines else Fore.RED + '✗ MISSING'}")
                print(f"  - relative_position: {Fore.GREEN + '✓' if has_relative_position else Fore.RED + '✗ MISSING'}")
                print(f"  - Range: {Fore.GREEN + '✓' if has_range else Fore.RED + '✗ MISSING'}")

                # 실제 값 출력
                if has_relative_position:
                    print(f"\n{Fore.CYAN}샘플 temporal metadata 값:")
                    print(f"  - line_number: {sample_metadata.get('line_number', 'N/A')}")
                    print(f"  - total_lines: {sample_metadata.get('total_lines', 'N/A')}")
                    print(f"  - relative_position: {sample_metadata.get('relative_position', 0.0):.4f}")
                    print(f"  - Range: {sample_metadata.get('Range', 'N/A')}")
                    print(f"{Fore.GREEN}✓ Temporal filtering 사용 가능")
                else:
                    print(f"\n{Fore.RED}⚠️ WARNING: Temporal metadata 필드가 없습니다!")
                    print(f"{Fore.RED}   → Temporal filtering이 정확하지 않을 수 있습니다.")
                    print(f"{Fore.YELLOW}   → FAISS 인덱스를 재구축하세요.")
                    print(f"\n{Fore.CYAN}Available fields: {list(sample_metadata.keys())}")
            else:
                print(f"{Fore.RED}⚠️ 샘플 문서를 찾을 수 없습니다.")
        except Exception as e:
            print(f"{Fore.RED}⚠️ 메타데이터 검증 중 오류: {e}")
        print(f"{Fore.CYAN}{'='*60}\n")

    def reload_index(self, new_index_path: str):
        """
        새로운 FAISS 인덱스를 로드 (BGE 모델은 유지)

        Args:
            new_index_path: 새 FAISS 인덱스 경로
        """
        from langchain_community.vectorstores import FAISS
        from langchain.embeddings.base import Embeddings

        # 커스텀 임베딩 클래스
        class CustomEmbeddings(Embeddings):
            def __init__(self, engine):
                self.engine = engine

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self.engine._encode_text(text, add_instruction=False) for text in texts]

            def embed_query(self, text: str) -> List[float]:
                return self.engine._encode_text(text, add_instruction=True)

        print(f"{Fore.YELLOW}🔄 인덱스 교체 중: {new_index_path}...")

        # 이전 인덱스 메모리 해제
        if hasattr(self, 'vectorstore'):
            del self.vectorstore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.index_path = new_index_path
        embeddings = CustomEmbeddings(self)
        self.vectorstore = FAISS.load_local(
            new_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"{Fore.GREEN}✓ 인덱스 교체 완료")

    def _encode_text(self, text: str, add_instruction: bool = False) -> List[float]:
        """텍스트를 임베딩으로 변환"""
        if add_instruction:
            text = self.query_instruction + text
        
        # 토크나이징
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # 인코딩
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs[0].mean(dim=1)  # Mean pooling
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings[0].cpu().numpy().tolist()
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        검색 실행 (PRE-FILTERING 지원)

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 최소 유사도 점수 (낮을수록 더 유사)
            filter: FAISS 메타데이터 필터 (PRE-FILTERING)
                   예: {"relative_position": {"$gte": 0.5, "$lte": 0.7}}

        Returns:
            검색 결과 리스트
        """
        # ✨ PRE-FILTERING: FAISS searches only within filter range BEFORE semantic search
        # However, LangChain FAISS uses POST-FILTERING (fetch_k → filter in Python)
        # So we need to set fetch_k high enough to get all candidates before filtering
        fetch_k = max(100, k * 3) if filter else k  # Fetch more when filtering

        results_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter,  # ← Filter is applied AFTER fetching fetch_k results
            fetch_k=fetch_k  # ← Fetch more results before filtering
        )

        # 결과 파싱
        search_results = []
        for rank, (doc, score) in enumerate(results_with_scores, 1):
            # 점수 필터링
            if score_threshold is not None and score > score_threshold:
                continue
            
            # SearchResult 객체 생성
            result = SearchResult(
                rank=rank,
                score=float(score),
                content=doc.page_content,
                metadata=doc.metadata,
                line_range=doc.metadata.get('Range', [0, 0]),
                time_start=doc.metadata.get('TimeStart', ''),
                time_end=doc.metadata.get('TimeEnd', '')
            )
            search_results.append(result)
        
        return search_results
    
    def search_with_reranking(
        self,
        query: str,
        k: int = 20,
        top_n: int = 5
    ) -> List[SearchResult]:
        """
        재순위화를 포함한 검색

        Args:
            query: 검색 쿼리
            k: 초기 검색 결과 수
            top_n: 재순위화 후 반환할 결과 수
        """
        # 초기 검색
        initial_results = self.search(query, k=k)

        # 여기에 재순위화 로직 추가 가능
        # 예: Cross-encoder, BM25 등

        return initial_results[:top_n]

# ── 결과 출력 함수들 ──────────────────────────────────────
def print_result(result: SearchResult, verbose: bool = False):
    """검색 결과를 보기 좋게 출력"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}[순위 {result.rank}] {Fore.GREEN}유사도: {result.score:.4f}")
    print(f"{Fore.MAGENTA}라인범위: {result.line_range}")
    print(f"{Fore.MAGENTA}시간: {result.time_start} - {result.time_end}")
    print(f"{Fore.CYAN}{'-'*80}")
    
    # 내용 출력 (길이 제한)
    content = result.content.strip()
    if not verbose and len(content) > 500:
        content = content[:500] + "..."
    
    # 줄바꿈으로 구분하여 출력
    lines = content.split('\n')
    for line in lines[:10] if not verbose else lines:  # verbose가 아니면 10줄까지만
        print(f"{Fore.WHITE}{line}")
    
    if not verbose and len(lines) > 10:
        print(f"{Fore.YELLOW}... ({len(lines)-10}줄 더 있음)")

def save_results(results: List[SearchResult], output_path: str, format: str = "json"):
    """검색 결과를 파일로 저장"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        data = []
        for r in results:
            data.append({
                "rank": r.rank,
                "score": r.score,
                "content": r.content,
                "line_range": r.line_range,
                "time_start": r.time_start,
                "time_end": r.time_end,
                "metadata": r.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    elif format == "csv":
        df = pd.DataFrame([{
            "rank": r.rank,
            "score": r.score,
            "content": r.content[:200],  # CSV는 내용 제한
            "line_range": r.line_range[0],
            "time_start": r.time_start,
            "time_end": r.time_end
        } for r in results])
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    elif format == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"Rank {r.rank} (Score: {r.score:.4f})\n")
                if r.line_range:
                    f.write(f"Lines: {r.line_range[0]} - {r.line_range[1]}\n")
                f.write(f"Time: {r.time_start} - {r.time_end}\n")
                f.write(f"Content:\n{r.content}\n")
                f.write("="*80 + "\n\n")
    
    print(f"{Fore.GREEN}✓ 결과 저장: {output_path}")

# ── 인터랙티브 검색 모드 ─────────────────────────────────
def interactive_search(engine: FAISSSearchEngine):
    """인터랙티브 검색 모드"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.YELLOW}인터랙티브 검색 모드")
    print(f"{Fore.WHITE}명령어: /quit (종료), /save (결과 저장), /clear (화면 정리)")
    print(f"       /top N (상위 N개 결과), /threshold F (유사도 임계값)")
    print(f"{Fore.CYAN}{'='*80}\n")
    
    # 설정
    top_k = 5
    threshold = None
    last_results = None
    
    while True:
        try:
            # 쿼리 입력
            query = input(f"\n{Fore.GREEN}검색어 입력: {Style.RESET_ALL}").strip()
            
            # 명령어 처리
            if query.startswith('/'):
                parts = query.split()
                cmd = parts[0].lower()
                
                if cmd == '/quit':
                    print(f"{Fore.YELLOW}검색 종료")
                    break
                
                elif cmd == '/clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                elif cmd == '/save' and last_results:
                    filename = input("저장할 파일명 (예: results.json): ").strip()
                    if filename:
                        format = filename.split('.')[-1] if '.' in filename else 'json'
                        save_results(last_results, filename, format)
                    continue
                
                elif cmd == '/top' and len(parts) > 1:
                    try:
                        top_k = int(parts[1])
                        print(f"{Fore.YELLOW}상위 {top_k}개 결과 표시")
                    except ValueError:
                        print(f"{Fore.RED}올바른 숫자를 입력하세요")
                    continue
                
                elif cmd == '/threshold' and len(parts) > 1:
                    try:
                        threshold = float(parts[1])
                        print(f"{Fore.YELLOW}유사도 임계값: {threshold}")
                    except ValueError:
                        print(f"{Fore.RED}올바른 숫자를 입력하세요")
                    continue
                
                else:
                    print(f"{Fore.RED}알 수 없는 명령어: {cmd}")
                    continue
            
            # 빈 쿼리 확인
            if not query:
                continue
            
            # 검색 실행
            print(f"\n{Fore.YELLOW}검색 중...")
            results = engine.search(query, k=top_k, score_threshold=threshold)
            last_results = results
            
            # 결과 출력
            if results:
                print(f"{Fore.GREEN}검색 결과: {len(results)}개")
                for result in results:
                    print_result(result, verbose=False)
            else:
                print(f"{Fore.RED}검색 결과가 없습니다.")
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}검색 종료")
            break
        except Exception as e:
            print(f"{Fore.RED}오류 발생: {e}")

# ── 배치 검색 모드 ───────────────────────────────────────
def batch_search(engine: FAISSSearchEngine, queries_file: str, output_dir: str):
    """여러 쿼리를 일괄 처리"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 쿼리 파일 읽기
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    print(f"{Fore.CYAN}배치 검색: {len(queries)}개 쿼리")
    
    all_results = {}
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] 검색: {query}")
        
        results = engine.search(query, k=5)
        all_results[query] = results
        
        # 개별 결과 저장
        query_safe = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_'))[:50]
        output_file = output_dir / f"result_{i:03d}_{query_safe}.json"
        save_results(results, str(output_file), format="json")
    
    # 전체 요약 저장
    summary_file = output_dir / "summary.json"
    summary_data = {
        query: [{
            "rank": r.rank,
            "score": r.score,
            "content_preview": r.content[:100]
        } for r in results[:3]]  # 상위 3개만
        for query, results in all_results.items()
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{Fore.GREEN}✓ 배치 검색 완료: {output_dir}")

# ── 메인 실행 함수 ───────────────────────────────────────
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="FAISS 검색 엔진")
    parser.add_argument(
        "--index-path",
        type=str,
        help="FAISS 인덱스 경로"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="검색 쿼리 (단일 검색)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="인터랙티브 모드"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="배치 검색용 쿼리 파일 경로"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="반환할 결과 수 (기본값: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="유사도 임계값"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="결과 저장 경로"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "txt"],
        default="json",
        help="출력 형식 (기본값: json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="사용할 디바이스 (cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    if not args.index_path:
        if os.name == 'nt':  # Windows
            args.index_path = r"C:\Users\soilf\lorin-proto\LORIN\make_faiss\log_faiss_index_bge_gemma2"
        else:  # Linux
            args.index_path = "/home/bigdata/lorin-proto/LORIN/make_faiss/log_faiss_index_bge_gemma2"
    
    # 검색 엔진 초기화
    print(f"{Fore.CYAN}FAISS 검색 엔진 초기화...")
    engine = FAISSSearchEngine(
        index_path=args.index_path,
        device=args.device
    )
    
    # 실행 모드 선택
    if args.interactive:
        # 인터랙티브 모드
        interactive_search(engine)
    
    elif args.batch:
        # 배치 모드
        output_dir = args.output or "batch_results"
        batch_search(engine, args.batch, output_dir)
    
    elif args.query:
        # 단일 쿼리 모드
        print(f"\n{Fore.YELLOW}검색: {args.query}")
        results = engine.search(
            args.query,
            k=args.top_k,
            score_threshold=args.threshold
        )
        
        # 결과 출력
        if results:
            print(f"{Fore.GREEN}검색 결과: {len(results)}개\n")
            for result in results:
                print_result(result, verbose=args.verbose)
            
            # 파일 저장
            if args.output:
                save_results(results, args.output, args.format)
        else:
            print(f"{Fore.RED}검색 결과가 없습니다.")
    
    else:
        # 기본: 인터랙티브 모드
        interactive_search(engine)

if __name__ == "__main__":
    main()