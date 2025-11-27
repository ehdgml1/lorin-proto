"""
Quality Evaluator Module - EXAONE-based Relevance Assessment
=============================================================

LORIN 시스템의 서브쿼리와 검색 문서 간 관련성을 EXAONE 모델로 평가하는 모듈입니다.

주요 특징:
- EXAONE 3.5-32B 모델 기반 관련성 평가
- Android 로그 데이터 특화 평가 프롬프트
- 배치 처리를 통한 토큰 효율성 최적화
- 구조화된 JSON 출력과 신뢰도 측정
- 에러 핸들링 및 폴백 메커니즘
- LangGraph State 호환성

아키텍처:
- QueryEvaluationResult: 쿼리별 종합 평가 결과
- DocumentEvaluation: 문서별 세부 평가 정보
- QualityEvaluator: 메인 평가 엔진
- quality_evaluator_node: LangGraph 통합 노드

통합:
- 입력: state.metadata.faiss_retriever 데이터
- 출력: state.metadata.quality_evaluator 네임스페이스
- 의존성: chatbot.py의 EXAONE 모델
"""

from __future__ import annotations
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import torch

from langchain_core.messages import AIMessage, BaseMessage

from ...llm.chatbot import Chatbot, LLMProvider
from ...make_faiss.search_engine import SearchResult
from ...logger.logger import get_logger
from ..state import AgentState
from ...prompt.template_loader import PromptTemplateLoader

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# Data Classes for Evaluation Results
# ─────────────────────────────────────────────────────────

@dataclass
class DocumentEvaluation:
    """개별 문서에 대한 관련성 평가 결과"""
    doc_rank: int
    content_preview: str  # 처음 200자
    relevance_score: float  # 0.0-1.0
    is_relevant: bool  # True/False 판단
    reasoning: str  # 평가 근거
    confidence: float  # 평가 신뢰도 0.0-1.0
    tokens_analyzed: int  # 분석된 토큰 수

@dataclass
class QueryEvaluationResult:
    """서브쿼리에 대한 종합 평가 결과"""
    query_id: str
    query_text: str
    is_relevant: bool  # 최종 관련성 판단
    relevance_score: float  # 0.0-1.0 종합 점수
    document_evaluations: List[DocumentEvaluation]
    evaluation_time: float  # 평가 소요 시간
    confidence: float  # 종합 신뢰도
    reasoning: str  # 종합 평가 근거

    # 성능 메트릭
    total_documents: int
    relevant_documents: int
    avg_doc_score: float
    max_doc_score: float
    tokens_used: int

    # replanner를 위한 상세 피드백 (default value 필드는 마지막에)
    improvement_suggestions: str = ""  # 개선 제안
    query_effectiveness: str = ""  # 쿼리 효과성 평가

    # 에러 정보
    error: Optional[str] = None
    fallback_used: bool = False

@dataclass
class EvaluationStats:
    """전체 평가 과정의 통계 정보"""
    total_queries: int
    successful_evaluations: int
    failed_evaluations: int
    total_documents_evaluated: int
    total_relevant_documents: int

    avg_relevance_score: float
    avg_confidence: float
    avg_evaluation_time: float

    total_tokens_used: int
    api_calls_made: int
    cache_hits: int
    fallback_count: int

    evaluation_time: float
    timestamp: str

# ─────────────────────────────────────────────────────────
# Korean Log Analysis Prompt Templates
# ─────────────────────────────────────────────────────────

class EvaluationPrompts:
    """EXAONE 모델을 위한 한국어 로그 분석 특화 프롬프트"""

    @staticmethod
    def get_relevance_evaluation_prompt(
        query: str,
        documents: List[SearchResult],
        max_docs: int = 5
    ) -> str:
        """관련성 평가를 위한 메인 프롬프트 생성 (Jinja2 template 사용)"""
        template_loader = PromptTemplateLoader()

        return template_loader.render_template(
            "quality_evaluation.j2",
            query=query,
            documents=documents[:max_docs],
            max_docs=max_docs
        )

    @staticmethod
    def get_simple_relevance_prompt(query: str, content: str) -> str:
        """단일 문서 간단 평가용 프롬프트"""
        return f"""다음 쿼리와 문서의 관련성을 0-100 점수로 평가하세요.

쿼리: {query}
문서: {content[:200]}...

점수만 숫자로 답하세요 (0-100):"""

# ─────────────────────────────────────────────────────────
# Quality Evaluator Implementation
# ─────────────────────────────────────────────────────────

class QualityEvaluator:
    """EXAONE 기반 관련성 평가 엔진"""

    def __init__(
        self,
        chatbot: Optional[Chatbot] = None,
        relevance_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        max_docs_per_query: int = 3,  # 5 → 3 (프롬프트 크기 감소)
        max_retries: int = 3,
        timeout_seconds: float = 120.0  # 30초 → 120초 (Multi-GPU 분산 고려)
    ):
        """
        Args:
            chatbot: EXAONE 챗봇 인스턴스 (None이면 자동 생성)
            relevance_threshold: 관련성 임계값 (>=이면 관련성 있음)
            confidence_threshold: 신뢰도 임계값
            max_docs_per_query: 쿼리당 최대 분석 문서 수
            max_retries: API 호출 재시도 횟수
            timeout_seconds: 평가 타임아웃
        """
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        self.max_docs_per_query = max_docs_per_query
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # 챗봇 초기화 (환경변수 기반)
        if chatbot is None:
            from ...utils import create_chatbot_from_env
            self.chatbot = create_chatbot_from_env(
                temperature=0.1  # 일관성을 위해 낮은 온도
                # max_tokens는 기본값(4096) 사용
            )
        else:
            self.chatbot = chatbot

        # 성능 추적
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.cache_hits = 0
        self.fallback_count = 0

        # 간단한 캐시 (쿼리-결과 해시)
        self.evaluation_cache: Dict[str, QueryEvaluationResult] = {}

        logger.info(f"[QualityEvaluator] Initialized with threshold={relevance_threshold}")

    def _generate_cache_key(self, query: str, doc_hashes: List[str]) -> str:
        """캐시 키 생성"""
        import hashlib
        cache_input = f"{query}:{':'.join(sorted(doc_hashes))}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def _get_document_hash(self, doc: SearchResult) -> str:
        """문서 해시 생성"""
        import hashlib
        content_key = f"{doc.content[:100]}:{doc.score}"
        return hashlib.md5(content_key.encode('utf-8')).hexdigest()[:8]

    async def _call_exaone_with_retry(self, prompt: str) -> str:
        """재시도 로직이 있는 EXAONE 호출 - Exponential backoff 적용"""
        import time
        last_error = None
        # 재시도 횟수 3→2로 감소 (성능 최적화)
        optimized_retries = min(self.max_retries, 2)

        for attempt in range(optimized_retries):
            call_start_time = time.time()
            try:
                self.total_api_calls += 1

                # 📊 LLM 호출 시작 로그
                prompt_length = len(prompt)
                logger.info(f"[QualityEvaluator] 🚀 LLM call #{self.total_api_calls} started (attempt {attempt + 1}/{optimized_retries}, prompt: {prompt_length} chars, timeout: {self.timeout_seconds}s)")

                # GPU 메모리 상태 확인 (호출 전)
                if torch.cuda.is_available():
                    free_memory_before = torch.cuda.mem_get_info(0)[0] / 1e9
                    allocated_memory_before = torch.cuda.memory_allocated(0) / 1e9
                    logger.debug(f"[QualityEvaluator] 💾 Pre-call GPU: Free={free_memory_before:.2f}GB, Allocated={allocated_memory_before:.2f}GB")

                    # 메모리 부족 경고
                    if free_memory_before < 2.0:
                        logger.warning(f"[QualityEvaluator] ⚠️ Low GPU memory before LLM call: {free_memory_before:.2f}GB (OOM risk)")

                # 타임아웃과 함께 호출
                response = await asyncio.wait_for(
                    self.chatbot.ask(prompt),
                    timeout=self.timeout_seconds
                )

                call_duration = time.time() - call_start_time

                # 토큰 계산 (근사치)
                estimated_tokens = len(prompt.split()) + len(response.split())
                self.total_tokens_used += estimated_tokens

                # 📊 LLM 호출 성공 로그
                response_length = len(response)
                logger.info(f"[QualityEvaluator] ✅ LLM call #{self.total_api_calls} succeeded in {call_duration:.2f}s (response: {response_length} chars, ~{estimated_tokens} tokens)")

                # GPU 메모리 상태 확인 (호출 후)
                if torch.cuda.is_available():
                    free_memory_after = torch.cuda.mem_get_info(0)[0] / 1e9
                    allocated_memory_after = torch.cuda.memory_allocated(0) / 1e9
                    memory_delta = allocated_memory_after - allocated_memory_before
                    logger.debug(f"[QualityEvaluator] 💾 Post-call GPU: Free={free_memory_after:.2f}GB, Allocated={allocated_memory_after:.2f}GB (Δ{memory_delta:+.2f}GB)")

                # GPU 메모리 정리 (CUDA OOM 방지)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug(f"[QualityEvaluator] 🧹 GPU cache cleared")

                return response

            except asyncio.TimeoutError:
                call_duration = time.time() - call_start_time
                last_error = f"Timeout after {self.timeout_seconds}s"
                logger.warning(f"[QualityEvaluator] ⏱️ LLM call #{self.total_api_calls} TIMEOUT after {call_duration:.2f}s (attempt {attempt + 1}/{optimized_retries})")

                # GPU 상태 확인 (타임아웃 시)
                if torch.cuda.is_available():
                    free_memory = torch.cuda.mem_get_info(0)[0] / 1e9
                    logger.warning(f"[QualityEvaluator] 💾 GPU memory at timeout: {free_memory:.2f}GB free")

            except RuntimeError as e:
                call_duration = time.time() - call_start_time
                # GPU OOM 에러 특별 처리
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(0) / 1e9
                        reserved = torch.cuda.memory_reserved(0) / 1e9
                        logger.error(f"[QualityEvaluator] 💥 GPU OOM during LLM call #{self.total_api_calls} (attempt {attempt + 1})")
                        logger.error(f"[QualityEvaluator] 💾 GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                    last_error = f"GPU Out of Memory: {str(e)}"
                else:
                    logger.error(f"[QualityEvaluator] 💥 RuntimeError in LLM call #{self.total_api_calls}: {e}")
                    last_error = f"RuntimeError: {str(e)}"

            except Exception as e:
                call_duration = time.time() - call_start_time
                last_error = str(e)
                logger.warning(f"[QualityEvaluator] ❌ LLM call #{self.total_api_calls} failed after {call_duration:.2f}s (attempt {attempt + 1}/{optimized_retries}): {type(e).__name__}: {e}")

            # 재시도 전 공격적인 메모리 정리
            if attempt < optimized_retries - 1:
                logger.info(f"[QualityEvaluator] 🧹 Performing aggressive memory cleanup before retry...")

                # 공격적인 메모리 정리
                import gc
                if torch.cuda.is_available():
                    # GPU 메모리 정리 (호출 순서 중요)
                    torch.cuda.empty_cache()      # 1. 캐시 비우기
                    torch.cuda.synchronize()      # 2. 모든 GPU 작업 완료 대기
                    gc.collect()                   # 3. Python 가비지 컬렉션

                    # 정리 후 메모리 상태 확인
                    free_after_cleanup = torch.cuda.mem_get_info(0)[0] / 1e9
                    allocated_after_cleanup = torch.cuda.memory_allocated(0) / 1e9
                    logger.info(f"[QualityEvaluator] 💾 After cleanup: Free={free_after_cleanup:.2f}GB, Allocated={allocated_after_cleanup:.2f}GB")

                # Exponential backoff: 2s, 4s, 8s... (1s → 2s로 증가, 메모리 해제 시간 확보)
                backoff_time = 2 ** (attempt + 1)
                logger.info(f"[QualityEvaluator] ⏳ Retrying in {backoff_time}s (allowing memory to stabilize)...")
                await asyncio.sleep(backoff_time)

        # 모든 재시도 실패
        logger.error(f"[QualityEvaluator] 💥 LLM call failed after {optimized_retries} attempts: {last_error}")
        raise Exception(f"EXAONE call failed after {optimized_retries} attempts: {last_error}")

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """EXAONE 응답을 JSON으로 파싱 (향상된 에러 처리)"""
        try:
            # 🔧 강화된 JSON 블록 추출
            json_str = self._extract_json_block(response)

            # 먼저 원본 JSON 파싱 시도 (가장 일반적인 경우)
            try:
                result = json.loads(json_str)
                logger.debug("[QualityEvaluator] Direct JSON parsing successful")
                return result
            except json.JSONDecodeError as parse_error:
                # 원본 파싱 실패시에만 정리 후 재시도
                logger.debug(f"[QualityEvaluator] Direct parsing failed: {parse_error}")
                logger.debug("[QualityEvaluator] Applying JSON sanitization")

                # 정리된 JSON으로 재시도
                sanitized_json = self._sanitize_json_string(json_str)
                return json.loads(sanitized_json)

        except json.JSONDecodeError as e:
            logger.error(f"[QualityEvaluator] JSON parsing failed: {e}")
            logger.error(f"[QualityEvaluator] Raw response: {response}")
            logger.error(f"[QualityEvaluator] Extracted JSON string: {json_str}")
            raise ValueError(f"Invalid JSON response: {e}")

    def _extract_json_block(self, response: str) -> str:
        """응답에서 JSON 블록을 추출하는 강화된 로직"""
        import re

        # 1. ```json 코드 블록 우선 시도
        json_pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
        match = re.search(json_pattern, response, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. ```없는 코드 블록 시도
        code_pattern = r'```\s*(\{[\s\S]*?\})\s*```'
        match = re.search(code_pattern, response, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # 3. 직접 JSON 객체 추출 (가장 큰 완전한 객체)
        if "{" in response and "}" in response:
            brace_positions = []
            brace_count = 0
            start_pos = -1

            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        # 완전한 JSON 객체 발견
                        candidate = response[start_pos:i+1]
                        brace_positions.append((len(candidate), candidate))

            # 가장 큰 JSON 객체 반환
            if brace_positions:
                brace_positions.sort(key=lambda x: x[0], reverse=True)
                return brace_positions[0][1].strip()

        # 4. 전체 응답이 JSON인 경우
        response = response.strip()
        if response.startswith('{') and response.endswith('}'):
            return response

        # 5. 실패한 경우 전체 응답 반환
        return response

    def _sanitize_json_string(self, json_str: str) -> str:
        """JSON 문자열에서 유효하지 않은 escape 문자와 제어 문자를 정리"""
        import re

        # 0. JSON 주석 제거 (LLM이 생성하는 주석 문제 해결)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        # 1. JSON 문자열 값 내의 제어 문자와 escape 처리
        def fix_string_content(match):
            string_content = match.group(1)

            # 제어 문자를 escape 시퀀스로 변환
            control_char_map = {
                '\n': '\\n',
                '\r': '\\r',
                '\t': '\\t',
                '\b': '\\b',
                '\f': '\\f',
            }

            # 제어 문자 처리 (이미 escape된 것은 제외)
            result = []
            i = 0
            while i < len(string_content):
                char = string_content[i]

                # 이미 올바르게 escape된 문자는 그대로 유지
                if char == '\\' and i + 1 < len(string_content):
                    next_char = string_content[i + 1]
                    if next_char in '"\\bfnrt/u':
                        result.append(char)
                        result.append(next_char)
                        i += 2
                        continue
                    else:
                        # 유효하지 않은 escape는 백슬래시를 escape
                        result.append('\\\\')
                        i += 1
                        continue

                # 제어 문자를 escape 시퀀스로 변환
                if char in control_char_map:
                    result.append(control_char_map[char])
                # ASCII 제어 문자 (0x00-0x1F, 0x7F-0x9F) 처리
                elif ord(char) < 0x20 or (0x7F <= ord(char) <= 0x9F):
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)

                i += 1

            return f'"{"".join(result)}"'

        # JSON 문자열 값들을 찾아서 처리 (개선된 패턴)
        # 이중 따옴표 내의 내용을 매칭하되, escape된 따옴표는 무시
        json_str = re.sub(r'"((?:[^"\\]|\\.)*)?"', fix_string_content, json_str)

        return json_str

    def _fallback_evaluation(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> QueryEvaluationResult:
        """폴백 평가 메커니즘 (LLM 평가 실패 시 기본값 제공)"""
        logger.warning(f"[QualityEvaluator] Using fallback evaluation for query: {query}")
        self.fallback_count += 1

        # LLM 판단 위임 - 단순 기본값만 제공 (문서별 평가 제거)
        doc_evaluations = []  # 빈 리스트
        avg_score = 0.1  # 낮은 기본값 (LLM 평가 실패 시)
        max_score = 0.1
        relevant_count = 0

        return QueryEvaluationResult(
            query_id="fallback",
            query_text=query,
            is_relevant=avg_score >= self.relevance_threshold,
            relevance_score=avg_score,
            document_evaluations=doc_evaluations,  # 빈 리스트
            evaluation_time=0.1,
            confidence=0.5,
            reasoning="LLM 평가 실패로 폴백 사용 - 낮은 신뢰도 기본값 적용",
            total_documents=len(documents),
            relevant_documents=relevant_count,  # 0
            avg_doc_score=avg_score,  # 0.1
            max_doc_score=max_score,  # 0.1
            tokens_used=0,
            improvement_suggestions="",  # 폴백 시 빈 값
            query_effectiveness="",  # 폴백 시 빈 값
            fallback_used=True
        )

    async def evaluate_query_document_relevance(
        self,
        subquery: Dict[str, Any],
        documents: List[SearchResult]
    ) -> QueryEvaluationResult:
        """서브쿼리와 문서들 간의 관련성 평가 (메인 함수)"""

        query_id = subquery.get("id", "unknown")
        query_text = subquery.get("question", subquery.get("text", "")).strip()

        # 디버깅: 평가하는 쿼리 정보 확인
        logger.info(f"🎯 [QualityEvaluator] Evaluating query:")
        logger.info(f"  - Query ID: {query_id}")
        logger.info(f"  - Query text (full): '{query_text}'")
        logger.info(f"  - Query length: {len(query_text)}")
        logger.info(f"  - Documents to evaluate: {len(documents)}")

        if not query_text:
            return QueryEvaluationResult(
                query_id=query_id,
                query_text=query_text,
                is_relevant=False,
                relevance_score=0.0,
                document_evaluations=[],
                evaluation_time=0.0,
                confidence=0.0,
                reasoning="빈 쿼리",
                total_documents=0,
                relevant_documents=0,
                avg_doc_score=0.0,
                max_doc_score=0.0,
                tokens_used=0,
                improvement_suggestions="",
                query_effectiveness="",
                error="Empty query text"
            )

        if not documents:
            return QueryEvaluationResult(
                query_id=query_id,
                query_text=query_text,
                is_relevant=False,
                relevance_score=0.0,
                document_evaluations=[],
                evaluation_time=0.0,
                confidence=0.0,
                reasoning="검색 결과 없음",
                total_documents=0,
                relevant_documents=0,
                avg_doc_score=0.0,
                max_doc_score=0.0,
                tokens_used=0,
                improvement_suggestions="",
                query_effectiveness="",
                error="No documents provided"
            )

        start_time = time.time()

        # 캐시 확인
        doc_hashes = [self._get_document_hash(doc) for doc in documents]
        cache_key = self._generate_cache_key(query_text, doc_hashes)

        if cache_key in self.evaluation_cache:
            cached_result = self.evaluation_cache[cache_key]
            cached_result.query_id = query_id  # ID 업데이트
            self.cache_hits += 1
            logger.debug(f"[QualityEvaluator] Cache hit for {query_id}")
            return cached_result

        # 문서 수 제한
        docs_to_evaluate = documents[:self.max_docs_per_query]

        # 🔄 Retry up to 3 times for LLM call + JSON parsing
        max_attempts = 3
        evaluation_data = None
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[QualityEvaluator] Attempt {attempt}/{max_attempts}: Evaluating {query_id}")

                # EXAONE 평가 프롬프트 생성
                prompt = EvaluationPrompts.get_relevance_evaluation_prompt(
                    query_text, docs_to_evaluate, self.max_docs_per_query
                )

                # EXAONE 호출
                response = await self._call_exaone_with_retry(prompt)

                # 응답 파싱
                evaluation_data = self._parse_evaluation_response(response)

                # JSON 파싱 성공
                logger.info(f"[QualityEvaluator] Attempt {attempt}/{max_attempts}: JSON parsing successful ✓")
                break  # Success - exit retry loop

            except Exception as e:
                last_error = str(e)
                logger.error(f"[QualityEvaluator] Attempt {attempt}/{max_attempts}: Failed - {e}")
                if attempt < max_attempts:
                    logger.info(f"[QualityEvaluator] Retrying evaluation for {query_id}...")
                    continue

        # Process result if successful
        if evaluation_data:
            try:
                logger.info(f"[QualityEvaluator] ✓ Successfully parsed JSON after {attempt} attempt(s)")

                # 결과 구조화
                result = self._structure_evaluation_result(
                    query_id, query_text, docs_to_evaluate, evaluation_data, start_time
                )

                # 캐시 저장
                if len(self.evaluation_cache) < 100:  # 캐시 크기 제한
                    self.evaluation_cache[cache_key] = result

                logger.info(f"[QualityEvaluator] Completed evaluation for {query_id}: "
                           f"relevant={result.is_relevant}, score={result.relevance_score:.3f}")

                return result

            except Exception as e:
                last_error = f"Result structuring failed: {str(e)}"
                logger.error(f"[QualityEvaluator] {last_error}")

        # All attempts failed
        logger.error(f"[QualityEvaluator] ✗ Failed to evaluate {query_id} after {max_attempts} attempts: {last_error}")

        try:
            # 폴백 평가 시도
            fallback_result = self._fallback_evaluation(query_text, docs_to_evaluate)
            fallback_result.query_id = query_id
            fallback_result.evaluation_time = time.time() - start_time
            fallback_result.error = f"EXAONE evaluation failed: {last_error}"
            return fallback_result

        except Exception as fallback_error:
            logger.error(f"[QualityEvaluator] Fallback evaluation also failed: {fallback_error}")

            return QueryEvaluationResult(
                query_id=query_id,
                query_text=query_text,
                is_relevant=False,
                relevance_score=0.0,
                document_evaluations=[],
                evaluation_time=time.time() - start_time,
                confidence=0.0,
                reasoning="평가 실패",
                total_documents=len(docs_to_evaluate),
                relevant_documents=0,
                avg_doc_score=0.0,
                max_doc_score=0.0,
                tokens_used=self.total_tokens_used,
                improvement_suggestions="",
                query_effectiveness="",
                error=f"Both EXAONE and fallback evaluation failed: {last_error}"
                )

    def _structure_evaluation_result(
        self,
        query_id: str,
        query_text: str,
        documents: List[SearchResult],
        evaluation_data: Dict[str, Any],
        start_time: float
    ) -> QueryEvaluationResult:
        """EXAONE 응답을 QueryEvaluationResult로 구조화 (문서별 평가 제거)"""

        overall = evaluation_data.get("overall_assessment", {})

        # 문서별 평가는 더 이상 생성하지 않음 (overall만 사용)
        doc_evaluations = []
        avg_doc_score = 0.0
        max_doc_score = 0.0
        relevant_count = 0

        evaluation_time = time.time() - start_time

        # Threshold-based relevance decision (override EXAONE's decision)
        relevance_score = overall.get("relevance_score", 0.0)
        is_relevant_by_threshold = relevance_score >= self.relevance_threshold

        return QueryEvaluationResult(
            query_id=query_id,
            query_text=query_text,
            is_relevant=is_relevant_by_threshold,  # Use threshold-based decision
            relevance_score=relevance_score,
            document_evaluations=doc_evaluations,  # 빈 리스트
            evaluation_time=evaluation_time,
            confidence=overall.get("confidence", 0.5),
            reasoning=overall.get("reasoning", ""),
            total_documents=len(documents),
            relevant_documents=relevant_count,  # 0
            avg_doc_score=avg_doc_score,  # 0.0
            max_doc_score=max_doc_score,  # 0.0
            tokens_used=self.total_tokens_used,
            improvement_suggestions=overall.get("improvement_suggestions", ""),  # replanner 개선 제안
            query_effectiveness=overall.get("query_effectiveness", "")  # 쿼리 효과성 평가
        )

    async def evaluate_batch_queries(
        self,
        subqueries_with_docs: List[Tuple[Dict[str, Any], List[SearchResult]]],
        max_concurrent: int = 5  # 2 → 5로 증가 (API 모델용)
    ) -> Tuple[Dict[str, QueryEvaluationResult], EvaluationStats]:
        """배치 쿼리 평가 (병렬 처리 - asyncio.gather 사용)

        Args:
            subqueries_with_docs: 평가할 (서브쿼리, 문서들) 튜플 리스트
            max_concurrent: 동시 실행 제한 (기본: 5개)
                           - API 모델 (Gemini/GPT/Llama): 5개 병렬 (빠름)
                           - 로컬 모델 (EXAONE): 1개 순차 (메모리 안전)
        """

        # 병렬 실행 전 강제 메모리 정리 (안전성 확보)
        if torch.cuda.is_available():
            import gc
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[QualityEvaluator] Pre-parallel memory cleanup completed")

        # 메모리 안전성 체크 및 동적 max_concurrent 조정
        free_memory_gb = 0
        if torch.cuda.is_available():
            free_memory_gb = torch.cuda.mem_get_info(0)[0] / 1e9

        # LLM provider에 따라 병렬도 조정
        if self.chatbot.provider.value in ["gemini", "gpt", "llama"]:
            # API 기반 모델: 병렬 처리 활성화 (GPU 메모리 불필요)
            max_concurrent = min(max_concurrent, 5)  # 최대 5개까지 병렬
            logger.info(f"[QualityEvaluator] Parallel execution mode for API model: max_concurrent={max_concurrent}")
        else:
            # 로컬 모델 (EXAONE): 메모리 안전을 위해 순차 실행
            max_concurrent = 1
            logger.info(f"[QualityEvaluator] Sequential execution mode for local model (max_concurrent=1)")

        logger.info(f"[QualityEvaluator] Starting PARALLEL evaluation of {len(subqueries_with_docs)} queries (max_concurrent={max_concurrent}, free_memory={free_memory_gb:.1f}GB)")
        start_time = time.time()

        results = {}
        successful_count = 0
        failed_count = 0
        total_docs = 0
        total_relevant_docs = 0
        all_relevance_scores = []
        all_confidences = []
        all_eval_times = []

        # 병렬 평가 실행 (청크 단위로 나누어 메모리 안전성 확보)
        total_queries = len(subqueries_with_docs)

        for chunk_idx in range(0, total_queries, max_concurrent):
            # 현재 청크 추출
            chunk = subqueries_with_docs[chunk_idx:chunk_idx + max_concurrent]
            chunk_size = len(chunk)

            logger.info(f"[QualityEvaluator] Processing chunk {chunk_idx//max_concurrent + 1} "
                       f"with {chunk_size} queries in parallel")

            # 병렬 실행할 태스크 생성
            tasks = []
            for subquery, documents in chunk:
                task = self.evaluate_query_document_relevance(subquery, documents)
                tasks.append(task)

            # 병렬 실행 (에러가 발생해도 다른 태스크는 계속 실행)
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 집계
            for idx, result in enumerate(chunk_results):
                query_id = chunk[idx][0].get("id", f"query_{chunk_idx + idx}")

                # 에러 체크
                if isinstance(result, Exception):
                    logger.error(f"[QualityEvaluator] Parallel evaluation failed for {query_id}: {result}")
                    failed_count += 1
                    continue

                # 정상 결과 처리
                if result.error:
                    failed_count += 1
                else:
                    successful_count += 1

                results[result.query_id] = result
                total_docs += result.total_documents
                total_relevant_docs += result.relevant_documents
                all_relevance_scores.append(result.relevance_score)
                all_confidences.append(result.confidence)
                all_eval_times.append(result.evaluation_time)

            # 각 청크 후 GPU 메모리 정리 (다음 청크를 위한 공간 확보)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"[QualityEvaluator] Memory cleanup after chunk {chunk_idx//max_concurrent + 1}")

        total_time = time.time() - start_time

        # 통계 계산
        stats = EvaluationStats(
            total_queries=len(subqueries_with_docs),
            successful_evaluations=successful_count,
            failed_evaluations=failed_count,
            total_documents_evaluated=total_docs,
            total_relevant_documents=total_relevant_docs,
            avg_relevance_score=sum(all_relevance_scores) / len(all_relevance_scores) if all_relevance_scores else 0.0,
            avg_confidence=sum(all_confidences) / len(all_confidences) if all_confidences else 0.0,
            avg_evaluation_time=sum(all_eval_times) / len(all_eval_times) if all_eval_times else 0.0,
            total_tokens_used=self.total_tokens_used,
            api_calls_made=self.total_api_calls,
            cache_hits=self.cache_hits,
            fallback_count=self.fallback_count,
            evaluation_time=total_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        logger.info(f"[QualityEvaluator] Batch evaluation completed in {total_time:.2f}s")
        logger.info(f"[QualityEvaluator] Success: {successful_count}/{len(subqueries_with_docs)}, "
                   f"Tokens: {self.total_tokens_used}, Cache hits: {self.cache_hits}")

        return results, stats

# ─────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────

__all__ = [
    "DocumentEvaluation",
    "QueryEvaluationResult",
    "EvaluationStats",
    "EvaluationPrompts",
    "QualityEvaluator"
]