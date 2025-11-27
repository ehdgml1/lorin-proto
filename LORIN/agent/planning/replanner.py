# LORIN/agent/replanner.py
"""
Replanner Module - Selective Subquery Reconstruction
====================================================
재계획 모듈: 실패한 서브쿼리만 선택적으로 재구성하는 시스템

주요 기능:
- 실패한 서브쿼리 분석 및 실패 원인 진단
- 성공한 서브쿼리는 보존하고 실패한 것만 재구성
- 의존성 관계 유지 및 수렴 보장
- 무한 루프 방지 및 품질 개선
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

from ...logger.logger import get_logger
from ..state import AgentState
from ..schema import (
    MetadataManager,
    create_replanner_metadata,
    get_subqueries
)
from ...prompt.template_loader import PromptTemplateLoader


@dataclass
class FailureAnalysis:
    """실패 분석 결과 - LLM이 자유롭게 분석"""
    subquery_id: str
    original_query: str
    failure_type: str  # LLM이 자유롭게 작성 (Enum 제거)
    confidence_score: float
    failure_reason: str
    suggested_improvements: List[str]
    context_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subquery_id": self.subquery_id,
            "original_query": self.original_query,
            "failure_type": self.failure_type,  # 이미 str이므로 .value 불필요
            "confidence_score": self.confidence_score,
            "failure_reason": self.failure_reason,
            "suggested_improvements": self.suggested_improvements,
            "context_hints": self.context_hints
        }


@dataclass
class ReconstructionResult:
    """재구성 결과"""
    subquery_id: str
    original_query: str
    reconstructed_query: str
    improvement_rationale: str
    confidence_score: float
    iteration_count: int
    failure_type: Optional[str] = None
    reconstruction_strategy: Optional[str] = None
    strategy_applied: Optional[str] = None
    improvements_applied: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subquery_id": self.subquery_id,
            "original_query": self.original_query,
            "reconstructed_query": self.reconstructed_query,
            "improvement_rationale": self.improvement_rationale,
            "confidence_score": self.confidence_score,
            "iteration_count": self.iteration_count,
            "failure_type": self.failure_type,
            "reconstruction_strategy": self.reconstruction_strategy,
            "strategy_applied": self.strategy_applied,
            "improvements_applied": self.improvements_applied or []
        }


class FailureAnalyzer:
    """실패 분석기 - 실패한 서브쿼리의 원인을 분석"""

    def __init__(self, chatbot, logger=None):
        self.chatbot = chatbot
        self.logger = logger or get_logger(__name__)

    def _get_user_question(self, state: AgentState) -> str:
        """상태에서 사용자 질문 추출"""
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        return "Unknown question"

    async def analyze_failures(
        self,
        state: AgentState,
        failed_subqueries: List[str]
    ) -> List[FailureAnalysis]:
        """실패한 서브쿼리들을 분석하여 실패 원인을 진단 (병렬 처리)"""

        # 품질 평가 결과 가져오기
        quality_data = MetadataManager.safe_get_metadata(state, "quality_evaluator")
        quality_results = quality_data.get("evaluation_results", {}) if quality_data else {}

        planner_data = MetadataManager.safe_get_metadata(state, "planner")
        planner_json = planner_data.get("last_plan_json", {}) if planner_data else {}

        # 병렬 분석을 위한 내부 함수
        async def analyze_single(subquery_id: str) -> FailureAnalysis:
            """단일 서브쿼리 분석 (병렬 실행용)"""
            try:
                return await self._analyze_single_failure(
                    subquery_id=subquery_id,
                    state=state,
                    quality_results=quality_results,
                    planner_data=planner_data
                )
            except Exception as e:
                self.logger.error(f"Failed to analyze subquery {subquery_id}: {e}")
                return self._create_default_analysis(subquery_id, state)

        # 🔧 GPU 메모리 기반 동시성 제한 (병렬 처리 안정성 확보)
        max_concurrent = 5  # 기본 동시 실행 수

        # GPU 메모리 여유 확인 및 동적 조정
        import torch
        if torch.cuda.is_available():
            free_memory_gb = torch.cuda.mem_get_info(0)[0] / 1e9
            if free_memory_gb < 4.0:
                self.logger.warning(f"[replanner] Low GPU memory ({free_memory_gb:.1f}GB), reducing max_concurrent to 2")
                max_concurrent = 2
            elif free_memory_gb < 6.0:
                self.logger.warning(f"[replanner] Moderate GPU memory ({free_memory_gb:.1f}GB), limiting max_concurrent to 3")
                max_concurrent = 3
            self.logger.info(f"[replanner] Parallel analysis with max_concurrent={max_concurrent} (free_memory={free_memory_gb:.1f}GB)")

        # Chunk 기반 병렬 처리
        all_analyses = []
        for chunk_idx in range(0, len(failed_subqueries), max_concurrent):
            chunk = failed_subqueries[chunk_idx:chunk_idx + max_concurrent]
            self.logger.info(f"[replanner] Analyzing chunk {chunk_idx//max_concurrent + 1}/{(len(failed_subqueries) + max_concurrent - 1)//max_concurrent} ({len(chunk)} queries)")

            tasks = [analyze_single(subquery_id) for subquery_id in chunk]
            chunk_analyses = await asyncio.gather(*tasks)
            all_analyses.extend(chunk_analyses)

            # 청크 간 메모리 정리
            if torch.cuda.is_available() and chunk_idx + max_concurrent < len(failed_subqueries):
                torch.cuda.empty_cache()

        return list(all_analyses)

    async def _analyze_single_failure(
        self,
        subquery_id: str,
        state: AgentState,
        quality_results: Dict,
        planner_data: Dict
    ) -> FailureAnalysis:
        """단일 서브쿼리 실패 분석"""

        # 서브쿼리 정보 추출
        planner_json = planner_data.get("last_plan_json", {})
        subqueries_list = planner_json.get("subqueries", [])

        # 리스트를 딕셔너리로 변환 (id를 키로 사용)
        subqueries = {sq.get("id"): sq for sq in subqueries_list if sq.get("id")}

        subquery_info = subqueries.get(subquery_id, {})
        original_query = subquery_info.get("text", "")

        # 품질 평가 결과 추출
        quality_info = quality_results.get(subquery_id, {})
        confidence_score = quality_info.get("confidence_score", 0.0)
        evaluation_details = quality_info.get("evaluation_details", "")

        # LLM을 사용한 실패 원인 분석
        analysis_prompt = self._build_failure_analysis_prompt(
            original_query=original_query,
            confidence_score=confidence_score,
            evaluation_details=evaluation_details,
            user_question=self._get_user_question(state),
            subquery_context=subquery_info
        )

        # 🔄 Retry up to 3 times for LLM call + JSON parsing
        max_attempts = 3
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"[FailureAnalyzer] Attempt {attempt}/{max_attempts}: Analyzing {subquery_id}")

                analysis_response = await self.chatbot.ask(analysis_prompt)

                # 응답 파싱
                failure_analysis = self._parse_analysis_response(
                    subquery_id=subquery_id,
                    original_query=original_query,
                    response=analysis_response,
                    confidence_score=confidence_score
                )

                # JSON 파싱이 성공했는지 확인 (failure_analysis가 유효한 데이터를 가지는지)
                if failure_analysis and failure_analysis.failure_type != "analysis_unavailable":
                    self.logger.info(f"[FailureAnalyzer] Attempt {attempt}/{max_attempts}: Analysis successful ✓")
                    return failure_analysis
                else:
                    self.logger.warning(f"[FailureAnalyzer] Attempt {attempt}/{max_attempts}: Got default analysis, retrying...")
                    if attempt < max_attempts:
                        continue

                return failure_analysis

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"[FailureAnalyzer] Attempt {attempt}/{max_attempts}: Failed - {e}")
                if attempt < max_attempts:
                    self.logger.info(f"[FailureAnalyzer] Retrying analysis for {subquery_id}...")
                    continue

        # All attempts failed
        self.logger.error(f"[FailureAnalyzer] ✗ Failed to analyze {subquery_id} after {max_attempts} attempts: {last_error}")
        return self._create_default_analysis(subquery_id, state)

    def _build_failure_analysis_prompt(
        self,
        original_query: str,
        confidence_score: float,
        evaluation_details,  # str 또는 Dict[str, str]
        user_question: str,
        subquery_context: Dict
    ) -> str:
        """실패 분석을 위한 프롬프트 구성 (Jinja2 template 사용)"""
        template_loader = PromptTemplateLoader()

        # evaluation_details가 딕셔너리이면 그대로 전달, 문자열이면 backward compatibility
        if isinstance(evaluation_details, dict):
            eval_details = evaluation_details
        else:
            # Backward compatibility: 문자열이면 reasoning만 있는 것으로 처리
            eval_details = {
                "reasoning": evaluation_details,
                "improvement_suggestions": "",
                "query_effectiveness": ""
            }

        return template_loader.render_template(
            "replanner_analysis.j2",
            user_question=user_question,
            subquery_context=subquery_context,
            original_query=original_query,
            confidence_score=confidence_score,
            evaluation_details=eval_details
        )

    def _extract_json_from_response(self, response: str) -> Optional[dict]:
        """planner와 동일한 방식으로 자연어 응답에서 JSON 추출"""
        if not response:
            return None

        import re

        # 1. 코드 블록에서 JSON 추출 시도
        json_block_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # 2. 첫 번째 '{' 부터 마지막 '}' 까지 추출 시도
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            blob = response[start : end + 1]
            return json.loads(blob)
        except json.JSONDecodeError:
            pass

        # 3. 정규표현식으로 JSON 객체 찾기
        json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _extract_info_with_regex(self, response: str) -> Dict[str, Any]:
        """응답에서 구조화된 분석 정보 추출 - 편향 없는 fallback 파싱"""
        import re

        # 기본값: 추정하지 않고 알 수 없음으로 처리
        result = {
            "failure_type": "relevance_too_low",  # 최소한의 기본값
            "failure_reason": "JSON 파싱 실패로 상세 분석 불가",
            "suggested_improvements": [],
            "context_hints": [],
            "confidence": 0.2  # 낮은 신뢰도로 설정
        }

        # 1. 실패 유형 추출 - 정확한 매칭
        failure_type_patterns = [
            r'"failure_type":\s*"([^"]+)"',
            r'failure_type[:\s]*([a-z_]+)',
        ]

        valid_failure_types = ["relevance_too_low", "coverage_insufficient", "specificity_lacking",
                              "context_mismatch", "ambiguity_high", "domain_misalignment"]

        for pattern in failure_type_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_type = match.group(1).strip().lower()
                if extracted_type in valid_failure_types:
                    result["failure_type"] = extracted_type
                    result["confidence"] = 0.6  # 정확한 매칭시 신뢰도 향상
                    break

        # 2. 실패 원인 추출 - 정확한 필드명 매칭
        reason_patterns = [
            r'"failure_reason":\s*"([^"]+)"',
            r'failure_reason[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
            r'(?:실패|오류)\s*(?:원인|이유)[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
            r'Reason[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
        ]

        for pattern in reason_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) >= 5:
                    result["failure_reason"] = extracted
                    break

        # 3. 개선 제안 추출 - 정확한 구조 매칭
        improvements_patterns = [
            r'"suggested_improvements":\s*\[(.*?)\]',
            r'suggested_improvements[:\s]*\[(.*?)\]',
            r'개선.*제안[:\s]*\[(.*?)\]',
        ]

        for pattern in improvements_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    items_text = match.group(1)
                    # JSON 배열 파싱
                    items = []
                    for item in re.findall(r'"([^"]+)"', items_text):
                        if len(item.strip()) > 3:
                            items.append(item.strip())
                    if items:
                        result["suggested_improvements"] = items[:5]
                        break
                except:
                    continue

        # 4. 컨텍스트 힌트 추출 - 정확한 구조 매칭
        hints_patterns = [
            r'"context_hints":\s*\[(.*?)\]',
            r'context_hints[:\s]*\[(.*?)\]',
            r'컨텍스트.*힌트[:\s]*\[(.*?)\]',
        ]

        for pattern in hints_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    items_text = match.group(1)
                    # JSON 배열 파싱
                    items = []
                    for item in re.findall(r'"([^"]+)"', items_text):
                        if len(item.strip()) > 2:
                            items.append(item.strip())
                    if items:
                        result["context_hints"] = items[:5]
                        break
                except:
                    continue

        # 5. 신뢰도 추출 - 정확한 필드명 매칭
        confidence_patterns = [
            r'"confidence":\s*([0-9]*\.?[0-9]+)',
            r'"confidence_score":\s*([0-9]*\.?[0-9]+)',
            r'confidence[:\s]*([0-9]*\.?[0-9]+)',
            r'신뢰도[:\s]*([0-9]*\.?[0-9]+)',
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    conf_value = float(match.group(1))
                    if 0.0 <= conf_value <= 1.0:
                        result["confidence"] = conf_value
                        break
                except ValueError:
                    continue

        # 6. 마지막 fallback: 구조화되지 않은 텍스트에서 최소한의 정보 추출
        # 개선 제안이 없는 경우에만 텍스트에서 추출 시도
        if not result["suggested_improvements"]:
            improvement_bullets = re.findall(r'[-•]\s*([^\n]{10,100})', response)
            if improvement_bullets:
                # 개선과 관련된 내용만 필터링
                meaningful_improvements = [
                    item.strip() for item in improvement_bullets
                    if any(keyword in item.lower() for keyword in ['키워드', '용어', '검색', '개선', '조정', 'keyword', 'improve', 'enhance'])
                ]
                if meaningful_improvements:
                    result["suggested_improvements"] = meaningful_improvements[:3]

        # 컨텍스트 힌트가 없는 경우에만 텍스트에서 추출 시도
        if not result["context_hints"]:
            hint_bullets = re.findall(r'[-•]\s*([^\n]{5,80})', response)
            if hint_bullets:
                # AOSP/Android 관련 내용만 필터링
                meaningful_hints = [
                    item.strip() for item in hint_bullets
                    if any(keyword in item.lower() for keyword in ['android', 'aosp', 'logcat', '태그', '시간', '필터', 'activity', 'service'])
                ]
                if meaningful_hints:
                    result["context_hints"] = meaningful_hints[:3]

        return result

    def _parse_analysis_response(
        self,
        subquery_id: str,
        original_query: str,
        response: str,
        confidence_score: float
    ) -> FailureAnalysis:
        """분석 응답 파싱 (planner와 동일한 안정적인 방식)"""

        # 1차: JSON 추출 시도
        analysis_data = self._extract_json_from_response(response)

        if analysis_data is None:
            # 2차: 정규표현식으로 정보 추출
            self.logger.info(f"JSON extraction failed for {subquery_id}, using regex extraction")
            analysis_data = self._extract_info_with_regex(response)

        try:
            # LLM이 자유롭게 작성한 failure_type 사용 (Enum 변환 제거)
            failure_type = analysis_data.get("failure_type", "analysis_unavailable")

            return FailureAnalysis(
                subquery_id=subquery_id,
                original_query=original_query,
                failure_type=failure_type,  # str 그대로 사용
                confidence_score=analysis_data.get("confidence", 0.5),
                failure_reason=analysis_data.get("failure_reason", "LLM 분석 실패"),
                suggested_improvements=analysis_data.get("suggested_improvements", []),
                context_hints=analysis_data.get("context_hints", [])
            )

        except Exception as e:
            self.logger.error(f"Error creating FailureAnalysis for {subquery_id}: {e}")
            return self._create_default_analysis(subquery_id, {"text": original_query})

    def _create_default_analysis(self, subquery_id: str, subquery_info: Dict[str, Any]) -> FailureAnalysis:
        """기본 분석 결과 생성 - LLM 분석 실패 시 최소한의 정보만 제공"""

        original_query = subquery_info.get("text", f"서브쿼리 {subquery_id}")

        return FailureAnalysis(
            subquery_id=subquery_id,
            original_query=original_query,
            failure_type="llm_analysis_failed",  # 하드코딩 제거, 사실 기반 정보
            confidence_score=0.0,  # 분석 실패이므로 신뢰도 0
            failure_reason="LLM 분석 실패로 인해 상세 분석 불가",
            suggested_improvements=[],  # 하드코딩된 제안 제거
            context_hints=[]  # 하드코딩된 힌트 제거
        )


class QueryRecomposer:
    """쿼리 재구성기 - 실패 분석을 바탕으로 서브쿼리를 재구성"""

    def __init__(self, chatbot, logger=None):
        self.chatbot = chatbot
        self.logger = logger or get_logger(__name__)

    def _extract_json_from_response(self, response: str) -> Optional[dict]:
        """LLM 응답에서 정확한 JSON 추출 - 편향 없는 파싱"""
        if not response:
            return None

        import re
        import json

        # 1. 가장 일반적인 패턴: ```json과 ``` 사이의 JSON
        json_code_patterns = [
            r'```json\s*(\{[\s\S]*?\})\s*```',
            r'```\s*(\{[\s\S]*?\})\s*```',
            r'```json\s*(\[[\s\S]*?\])\s*```',  # 배열 형태도 지원
        ]

        for pattern in json_code_patterns:
            matches = re.finditer(pattern, response, re.MULTILINE)
            for match in matches:
                json_candidate = match.group(1)
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    # JSON 문법 오류가 있을 수 있으므로 정리 시도
                    cleaned = self._clean_json_string(json_candidate)
                    if cleaned:
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            continue

        # 2. 균형잡힌 중괄호 블록 찾기 (가장 완전한 JSON 우선)
        json_candidates = self._find_balanced_json_blocks(response)

        # 크기 순으로 정렬 (큰 것부터 시도)
        json_candidates.sort(key=len, reverse=True)

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                # 유효한 구조인지 검증
                if self._is_valid_response_structure(parsed):
                    return parsed
            except json.JSONDecodeError:
                # JSON 정리 후 재시도
                cleaned = self._clean_json_string(candidate)
                if cleaned:
                    try:
                        parsed = json.loads(cleaned)
                        if self._is_valid_response_structure(parsed):
                            return parsed
                    except json.JSONDecodeError:
                        continue

        return None

    def _clean_json_string(self, json_str: str) -> Optional[str]:
        """JSON 문자열 정리 - 일반적인 LLM 출력 문제 해결"""
        import re

        if not json_str:
            return None

        # 앞뒤 공백 제거
        cleaned = json_str.strip()

        # 일반적인 마크다운 아티팩트 제거
        cleaned = re.sub(r'^```[a-z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)

        # 주석 제거 (// 또는 /* */)
        cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

        # 후행 쉼표 제거
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

        # 잘못된 따옴표 수정 (곱슬따옴표 등)
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")

        return cleaned.strip()

    def _find_balanced_json_blocks(self, text: str) -> List[str]:
        """텍스트에서 균형잡힌 JSON 블록들 찾기"""
        import re

        candidates = []

        # 모든 { 위치 찾기
        for match in re.finditer(r'\{', text):
            start_pos = match.start()
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(text[start_pos:], start_pos):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = text[start_pos:i+1]
                            if len(candidate) > 10:  # 최소 크기 필터
                                candidates.append(candidate)
                            break

        return candidates

    def _is_valid_response_structure(self, parsed_json: dict) -> bool:
        """응답 구조가 유효한지 검증"""
        if not isinstance(parsed_json, dict):
            return False

        # reconstruction 응답 구조 검증 (새로운 템플릿 기준)
        if "reconstructed_queries" in parsed_json:
            queries = parsed_json["reconstructed_queries"]
            if isinstance(queries, list) and len(queries) > 0:
                # 첫 번째 항목이 올바른 구조인지 확인
                first_query = queries[0]
                if isinstance(first_query, dict):
                    # 새로운 템플릿 필드 확인
                    required_fields = ["original_id", "reconstructed_text"]
                    has_required = all(field in first_query for field in required_fields)

                    # 플레이스홀더 검증
                    reconstructed_text = first_query.get("reconstructed_text", "")
                    has_placeholders = any(placeholder in reconstructed_text for placeholder in [
                        "WRITE_IMPROVED_QUERY_HERE",
                        "APPLY_STRATEGY_SPECIFIC_IMPROVEMENTS_HERE",
                        "EXPLAIN_WHY_BETTER"
                    ])

                    return has_required and not has_placeholders

        # analysis 응답 구조 검증
        required_analysis_fields = ["failure_type", "failure_reason", "suggested_improvements"]
        if any(field in parsed_json for field in required_analysis_fields):
            return True

        # 기본적인 JSON 객체라면 허용
        return len(parsed_json) > 0

    def _get_user_question(self, state: AgentState) -> str:
        """상태에서 사용자 질문 추출"""
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        return "Unknown question"

    async def reconstruct_queries(
        self,
        state: AgentState,
        failure_analyses: List[FailureAnalysis],
        iteration_count: int = 1
    ) -> List[ReconstructionResult]:
        """실패 분석을 바탕으로 서브쿼리들을 재구성"""

        reconstructions = []

        # 성공한 서브쿼리 정보 수집 (컨텍스트로 활용)
        successful_context = self._collect_successful_context(state)
        self.logger.info(f"[replanner] Using {len(successful_context.get('successful_queries', []))} successful queries as context")

        for analysis in failure_analyses:
            try:
                reconstruction = await self._reconstruct_single_query(
                    analysis=analysis,
                    state=state,
                    successful_context=successful_context,
                    iteration_count=iteration_count
                )
                reconstructions.append(reconstruction)

            except Exception as e:
                self.logger.error(f"Failed to reconstruct query {analysis.subquery_id}: {e}")
                # 기본 재구성 결과 생성
                reconstructions.append(self._create_default_reconstruction(analysis, iteration_count))

        return reconstructions

    def _collect_successful_context(self, state: AgentState) -> Dict[str, Any]:
        """성공한 서브쿼리들의 컨텍스트 정보 수집"""

        quality_data = MetadataManager.safe_get_metadata(state, "quality_evaluator")
        quality_results = quality_data.get("evaluation_results", {}) if quality_data else {}

        planner_data = MetadataManager.safe_get_metadata(state, "planner")
        planner_json = planner_data.get("last_plan_json", {}) if planner_data else {}

        successful_queries = []
        successful_patterns = []

        # 리스트를 딕셔너리로 변환
        subqueries_list = planner_json.get("subqueries", [])
        subqueries_dict = {sq.get("id"): sq for sq in subqueries_list if sq.get("id")}

        for subquery_id, quality_info in quality_results.items():
            if quality_info.get("is_relevant", False):
                subquery_info = subqueries_dict.get(subquery_id, {})
                if subquery_info:
                    successful_queries.append({
                        "id": subquery_id,
                        "question": subquery_info.get("text", ""),
                        "score": quality_info.get("confidence_score", 0.0)
                    })

        # 성공 패턴 추출
        if successful_queries:
            high_scoring = [q for q in successful_queries if q["score"] > 0.7]
            successful_patterns = [q["question"] for q in high_scoring[:3]]

        return {
            "successful_queries": successful_queries,
            "successful_patterns": successful_patterns,
            "total_successful": len(successful_queries)
        }

    async def _reconstruct_single_query(
        self,
        analysis: FailureAnalysis,
        state: AgentState,
        successful_context: Dict[str, Any],
        iteration_count: int
    ) -> ReconstructionResult:
        """단일 서브쿼리 재구성"""

        # 재구성 프롬프트 구성
        reconstruction_prompt = self._build_reconstruction_prompt(
            analysis=analysis,
            user_question=self._get_user_question(state),
            successful_context=successful_context,
            iteration_count=iteration_count
        )

        # 🔄 Retry up to 3 times for LLM call + JSON parsing
        max_attempts = 3
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"[QueryRecomposer] Attempt {attempt}/{max_attempts}: Reconstructing {analysis.subquery_id}")

                reconstruction_response = await self.chatbot.ask(reconstruction_prompt)

                # 응답 파싱
                reconstruction_result = self._parse_reconstruction_response(
                    analysis=analysis,
                    response=reconstruction_response,
                    iteration_count=iteration_count
                )

                # JSON 파싱이 성공했는지 확인 (유효한 재구성 쿼리가 있는지)
                if reconstruction_result and reconstruction_result.reconstructed_query != analysis.original_query:
                    self.logger.info(f"[QueryRecomposer] Attempt {attempt}/{max_attempts}: Reconstruction successful ✓")
                    return reconstruction_result
                else:
                    self.logger.warning(f"[QueryRecomposer] Attempt {attempt}/{max_attempts}: Got same query, retrying...")
                    if attempt < max_attempts:
                        continue

                return reconstruction_result

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"[QueryRecomposer] Attempt {attempt}/{max_attempts}: Failed - {e}")
                if attempt < max_attempts:
                    self.logger.info(f"[QueryRecomposer] Retrying reconstruction for {analysis.subquery_id}...")
                    continue

        # All attempts failed
        self.logger.error(f"[QueryRecomposer] ✗ Failed to reconstruct {analysis.subquery_id} after {max_attempts} attempts: {last_error}")
        return self._create_default_reconstruction(analysis, iteration_count)

    def _build_reconstruction_prompt(
        self,
        analysis: FailureAnalysis,
        user_question: str,
        successful_context: Dict[str, Any],
        iteration_count: int
    ) -> str:
        """재구성을 위한 프롬프트 구성 (Jinja2 template 사용)"""
        template_loader = PromptTemplateLoader()

        # Convert single analysis to list format expected by template
        failure_analyses = [{
            "subquery_id": analysis.subquery_id,
            "original_query": analysis.original_query,
            "failure_type": analysis.failure_type,  # 이미 str
            "failure_reason": analysis.failure_reason,
            "suggested_improvements": analysis.suggested_improvements
        }]

        return template_loader.render_template(
            "replanner_reconstruction.j2",
            target_query=analysis.original_query,
            user_question=user_question,
            successful_context=successful_context,
            failure_analyses=failure_analyses
        )

    def _extract_reconstruction_info_with_regex(self, response: str, original_query: str) -> Dict[str, Any]:
        """응답에서 구조화된 정보 추출 - 편향 없는 fallback 파싱"""
        import re

        # 기본값: 원본 쿼리 유지 (편향 없음)
        result = {
            "reconstructed_query": original_query,
            "improvement_rationale": "JSON 파싱 실패로 원본 쿼리 유지",
            "confidence": 0.3,
            "key_changes": [],
            "failure_type": None,
            "reconstruction_strategy": None,
            "strategy_applied": None,
            "improvements_applied": []
        }

        # 🔧 플레이스홀더 텍스트 감지 및 처리 (확장됨)
        placeholder_patterns = [
            r'WRITE_YOUR_IMPROVED_QUERY_HERE',
            r'EXPLAIN_YOUR_IMPROVEMENTS_HERE',
            r'\[WRITE_IMPROVED_QUERY_HERE\]',
            r'APPLY_STRATEGY_SPECIFIC_IMPROVEMENTS_HERE',
            r'\[EXPLAIN_WHY_BETTER\]',
            r'STRATEGY_SPECIFIC_IMPROVEMENT_\d+',
            r'YOUR_SEMANTIC_KEYWORD_SEQUENCE',  # 템플릿 플레이스홀더 추가
            r'YOUR_EXPLANATION',                # 템플릿 플레이스홀더 추가
            r'YOUR_.*?_HERE',                   # 포괄적 YOUR_*_HERE 패턴
            r'WRITE_.*?_HERE',                  # 포괄적 WRITE_*_HERE 패턴
            r'\[YOUR_.*?\]',                    # 대괄호 포함 YOUR 패턴
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                self.logger.warning(f"Detected placeholder in LLM response: {pattern}")
                result["improvement_rationale"] = "LLM 응답에 플레이스홀더 포함, 원본 쿼리 유지"
                result["confidence"] = 0.1  # 매우 낮은 신뢰도
                return result

        # 1. 재구성된 쿼리 추출 - 새로운 템플릿 필드 지원
        query_extraction_patterns = [
            # 새로운 템플릿 필드 (우선순위)
            r'"reconstructed_text":\s*"([^"]+)"',
            # 호환성을 위한 기존 필드
            r'"reconstructed_query":\s*"([^"]+)"',
            # 구조화된 텍스트 패턴
            r'(?:재구성된?|개선된?|새로운)\s*(?:쿼리|질문)[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
            r'Reconstructed[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
            r'Improved[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
        ]

        for pattern in query_extraction_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                extracted = match.group(1).strip()
                # 유효성 검증: 최소 길이와 원본과 다른지 확인
                if len(extracted) >= 5 and extracted != original_query:
                    result["reconstructed_query"] = extracted
                    result["confidence"] = 0.7  # 실제 추출 성공시 신뢰도 향상
                    break
            if result["reconstructed_query"] != original_query:
                break

        # 2. 개선 논리 추출 - 정확한 필드명 매칭
        rationale_patterns = [
            r'"improvement_rationale":\s*"([^"]+)"',
            r'"rationale":\s*"([^"]+)"',
            r'(?:개선|향상)\s*(?:논리|이유|근거)[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
            r'Rationale[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
        ]

        for pattern in rationale_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) >= 5:
                    result["improvement_rationale"] = extracted
                    break

        # 3. 신뢰도 추출 - 정확한 필드명 매칭
        confidence_patterns = [
            r'"confidence":\s*([0-9]*\.?[0-9]+)',
            r'confidence[:\s]*([0-9]*\.?[0-9]+)',
            r'신뢰도[:\s]*([0-9]*\.?[0-9]+)',
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    conf_value = float(match.group(1))
                    if 0.0 <= conf_value <= 1.0:
                        result["confidence"] = conf_value
                        break
                except ValueError:
                    continue

        # 4. 개선사항 추출 - 정확한 구조 매칭
        improvements_patterns = [
            r'"improvements_applied":\s*\[(.*?)\]',
            r'"key_changes":\s*\[(.*?)\]',
            r'개선.*사항[:\s]*\[(.*?)\]',
        ]

        for pattern in improvements_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    items_text = match.group(1)
                    # 간단한 배열 파싱
                    items = []
                    for item in re.findall(r'"([^"]+)"', items_text):
                        if len(item.strip()) > 2:
                            items.append(item.strip())
                    if items:
                        result["key_changes"] = items[:3]
                        break
                except:
                    continue

        # 5. 새로운 필드들 추출
        # failure_type 추출
        failure_type_patterns = [
            r'"failure_type":\s*"([^"]+)"',
            r'failure_type[:\s]*([a-z_]+)',
        ]

        for pattern in failure_type_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().strip('"\'')
                if extracted in ["relevance_too_low", "coverage_insufficient", "specificity_lacking",
                               "context_mismatch", "ambiguity_high", "domain_misalignment"]:
                    result["failure_type"] = extracted
                    break

        # reconstruction_strategy 추출
        strategy_patterns = [
            r'"reconstruction_strategy":\s*"([^"]+)"',
            r'reconstruction_strategy[:\s]*([a-z_]+)',
        ]

        for pattern in strategy_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().strip('"\'')
                if extracted in ["keyword_enhancement", "scope_expansion", "precision_increase",
                               "context_realignment", "disambiguation", "domain_realignment", "general_enhancement"]:
                    result["reconstruction_strategy"] = extracted
                    break

        # strategy_applied 추출
        strategy_applied_patterns = [
            r'"strategy_applied":\s*"([^"]+)"',
            r'strategy_applied[:\s]*(?:"|\')?([^\n"\']+)(?:"|\')?',
        ]

        for pattern in strategy_applied_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip().strip('"\'')
                if len(extracted) >= 10:
                    result["strategy_applied"] = extracted
                    break

        # improvements_applied 배열 추출
        if not result["improvements_applied"]:
            improvements_array_patterns = [
                r'"improvements_applied":\s*\[(.*?)\]',
                r'improvements_applied[:\s]*\[(.*?)\]',
            ]

            for pattern in improvements_array_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        items_text = match.group(1)
                        items = [item.strip().strip('"\'') for item in re.findall(r'"([^"]+)"', items_text)]
                        if items and all(len(item) > 3 for item in items):
                            result["improvements_applied"] = items[:3]
                            break
                    except:
                        continue

        # 6. 마지막 fallback: 구조화되지 않은 텍스트에서 변경사항 추출
        if not result["key_changes"]:
            changes = re.findall(r'[-•]\s*([^\n]{5,80})', response)
            if changes:
                # 의미있는 변경사항만 필터링
                meaningful_changes = [
                    change.strip() for change in changes
                    if any(keyword in change.lower() for keyword in ['추가', '개선', '변경', 'add', 'improve', 'enhance'])
                ]
                if meaningful_changes:
                    result["key_changes"] = meaningful_changes[:3]

        # 기존 key_changes를 improvements_applied로 복사 (호환성)
        if not result["improvements_applied"] and result["key_changes"]:
            result["improvements_applied"] = result["key_changes"]

        return result

    def _parse_reconstruction_response(
        self,
        analysis: FailureAnalysis,
        response: str,
        iteration_count: int
    ) -> ReconstructionResult:
        """재구성 응답 파싱 (planner와 동일한 안정적인 방식)"""

        # 디버깅: LLM 응답 로깅
        self.logger.info(
            f"🤖 [{analysis.subquery_id}] LLM Reconstruction Response:\n"
            f"  Original Query: '{analysis.original_query}'\n"
            f"  LLM Response (first 500 chars): {response[:500]}"
        )

        # 1차: JSON 추출 시도
        reconstruction_data = self._extract_json_from_response(response)

        if reconstruction_data is None:
            # 2차: 정규표현식으로 정보 추출
            self.logger.info(f"JSON extraction failed for reconstruction {analysis.subquery_id}, using regex extraction")
            reconstruction_data = self._extract_reconstruction_info_with_regex(response, analysis.original_query)
        else:
            # JSON 파싱 성공 - 배열 구조 처리
            if "reconstructed_queries" in reconstruction_data:
                # LLM이 배열 형태로 반환한 경우 첫 번째 항목 추출
                queries_array = reconstruction_data.get("reconstructed_queries", [])
                if queries_array and isinstance(queries_array, list) and len(queries_array) > 0:
                    reconstruction_data = queries_array[0]
                    self.logger.info(f"✅ [{analysis.subquery_id}] Extracted first item from reconstructed_queries array")

        # 디버깅: 파싱 결과 로깅
        reconstructed_text = reconstruction_data.get("reconstructed_text") or reconstruction_data.get("reconstructed_query", "")
        self.logger.info(
            f"📝 [{analysis.subquery_id}] Parsed Reconstruction Data:\n"
            f"  reconstructed_text: '{reconstructed_text}'\n"
            f"  Original for comparison: '{analysis.original_query}'\n"
            f"  Are they same? {reconstructed_text == analysis.original_query}"
        )

        try:
            return ReconstructionResult(
                subquery_id=analysis.subquery_id,
                original_query=analysis.original_query,
                reconstructed_query=reconstructed_text or analysis.original_query,
                improvement_rationale=reconstruction_data.get("improvement_rationale", "재구성 논리 누락"),
                confidence_score=reconstruction_data.get("confidence", 0.5),
                iteration_count=iteration_count,
                failure_type=reconstruction_data.get("failure_type", analysis.failure_type),  # 이미 str
                reconstruction_strategy=reconstruction_data.get("reconstruction_strategy"),
                strategy_applied=reconstruction_data.get("strategy_applied"),
                improvements_applied=reconstruction_data.get("improvements_applied", [])
            )

        except Exception as e:
            self.logger.error(f"Error creating ReconstructionResult for {analysis.subquery_id}: {e}")
            return self._create_default_reconstruction(analysis, iteration_count)

    def _create_default_reconstruction(
        self,
        analysis: FailureAnalysis,
        iteration_count: int
    ) -> ReconstructionResult:
        """기본 재구성 결과 생성"""

        # 원본 쿼리 유지 (편향 없는 기본값)
        original = analysis.original_query

        return ReconstructionResult(
            subquery_id=analysis.subquery_id,
            original_query=original,
            reconstructed_query=original,  # 편향 제거: 원본 유지
            improvement_rationale="재구성 실패로 원본 쿼리 유지",
            confidence_score=0.2,  # 낮은 신뢰도
            iteration_count=iteration_count,
            failure_type=analysis.failure_type,  # 이미 str
            reconstruction_strategy="general_enhancement",
            strategy_applied="기본 전략 - 재구성 실패",
            improvements_applied=[]
        )


class ConvergenceDetector:
    """수렴 감지기 - 무한 루프 방지 및 수렴 보장"""

    def __init__(self, max_iterations: int = 25, min_improvement_threshold: float = 0.01):
        self.max_iterations = max_iterations
        self.min_improvement_threshold = min_improvement_threshold
        self.logger = get_logger(__name__)

    def should_continue_iteration(
        self,
        current_iteration: int,
        iteration_history: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """반복을 계속할지 결정"""

        # 최대 반복 횟수 확인
        if current_iteration >= self.max_iterations:
            return False, f"최대 반복 횟수 {self.max_iterations}에 도달"

        # 이전 반복과의 개선도 확인 (정보 로깅만, 중단하지 않음)
        if len(iteration_history) >= 2:
            prev_scores = iteration_history[-2].get("average_score", 0.0)
            curr_scores = iteration_history[-1].get("average_score", 0.0)
            improvement = curr_scores - prev_scores

            # 🔧 개선도가 낮아도 품질 통과할 때까지 계속 진행
            if improvement < self.min_improvement_threshold:
                self.logger.info(f"개선도 {improvement:.3f}이 임계값 {self.min_improvement_threshold} 미만이지만 계속 진행")
                # return False 제거 - 중단하지 않음

        # 연속 실패 확인 (매우 관대하게)
        if len(iteration_history) >= 12:
            recent_failures = [
                hist.get("failed_count", 0)
                for hist in iteration_history[-12:]
            ]
            if all(count > 0 for count in recent_failures):
                return False, "12회 연속 실패 서브쿼리 존재"

        return True, "반복 계속 가능"

    def analyze_convergence_pattern(
        self,
        iteration_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """수렴 패턴 분석"""

        if not iteration_history:
            return {"pattern": "insufficient_data", "confidence": 0.0}

        # 점수 개선 패턴 분석
        scores = [hist.get("average_score", 0.0) for hist in iteration_history]

        if len(scores) < 2:
            return {"pattern": "insufficient_data", "confidence": 0.0}

        # 개선 추세 계산
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]

        # 패턴 분류
        if all(imp >= 0 for imp in improvements):
            pattern = "improving"
            confidence = min(0.9, sum(improvements) / len(improvements) * 2)
        elif all(imp <= 0 for imp in improvements[-2:]):
            pattern = "degrading"
            confidence = 0.7
        elif abs(improvements[-1]) < 0.05:
            pattern = "converged"
            confidence = 0.8
        else:
            pattern = "fluctuating"
            confidence = 0.5

        return {
            "pattern": pattern,
            "confidence": confidence,
            "latest_improvement": improvements[-1] if improvements else 0.0,
            "average_improvement": sum(improvements) / len(improvements) if improvements else 0.0
        }


async def replanning_node(chatbot, agent_name: str = "replanner"):
    """
    재계획 노드 - 실패한 서브쿼리만 선택적으로 재구성

    Args:
        chatbot: LLM 인스턴스
        agent_name: 에이전트 이름

    Returns:
        재계획 노드 함수
    """
    logger = get_logger(__name__)

    # 컴포넌트 초기화
    failure_analyzer = FailureAnalyzer(chatbot, logger)
    query_recomposer = QueryRecomposer(chatbot, logger)
    convergence_detector = ConvergenceDetector()

    async def replanning_node_impl(state: AgentState) -> AgentState:
        """재계획 노드 구현"""

        try:
            logger.info(f"[{agent_name}] 재계획 프로세스 시작")

            # 실패한 서브쿼리 식별
            failed_subqueries = _identify_failed_subqueries(state)

            if not failed_subqueries:
                logger.info(f"[{agent_name}] 실패한 서브쿼리 없음, 재계획 불필요")
                return type(state)(
                    messages=state.get("messages", []),
                    context_messages=state.get("context_messages", []),
                    metadata=state.get("metadata", {}),
                    current_agent=agent_name,
                    session_id=state.get("session_id", "")
                )

            logger.info(f"[{agent_name}] 실패한 서브쿼리 {len(failed_subqueries)}개 발견: {failed_subqueries}")

            # 반복 정보 초기화
            replanner_data = MetadataManager.safe_get_metadata(state, "replanner")
            iteration_count = (replanner_data.get("iteration_count", 0) if replanner_data else 0) + 1

            # 수렴 조건 확인
            iteration_history = replanner_data.get("iteration_history", []) if replanner_data else []
            should_continue, reason = convergence_detector.should_continue_iteration(
                iteration_count, iteration_history
            )

            if not should_continue:
                logger.warning(f"[{agent_name}] 반복 중단: {reason}")
                _update_replanner_metadata(state, {
                    "status": "converged",
                    "convergence_reason": reason,
                    "iteration_count": iteration_count
                })

                # 🔧 수정: 업데이트된 metadata를 명시적으로 전달
                updated_metadata = state.get("metadata", {})

                return type(state)(
                    messages=state.get("messages", []),
                    context_messages=state.get("context_messages", []),
                    metadata=updated_metadata,  # ← 업데이트된 metadata 사용
                    current_agent=agent_name,
                    session_id=state.get("session_id", "")
                )

            # 1단계: 실패 분석
            logger.info(f"[{agent_name}] 1단계: 실패 분석 수행")
            failure_analyses = await failure_analyzer.analyze_failures(state, failed_subqueries)

            # 2단계: 쿼리 재구성
            logger.info(f"[{agent_name}] 2단계: 쿼리 재구성 수행")
            reconstructions = await query_recomposer.reconstruct_queries(
                state, failure_analyses, iteration_count
            )

            # 3단계: 재구성된 서브쿼리로 플래너 데이터 업데이트
            logger.info(f"[{agent_name}] 3단계: 플래너 데이터 업데이트")
            _update_planner_with_reconstructions(state, reconstructions)

            # 4단계: 메타데이터 업데이트
            _update_replanner_metadata(state, {
                "status": "completed",
                "iteration_count": iteration_count,
                "failed_count": len(failed_subqueries),
                "reconstructed_count": len(reconstructions),
                "failure_analyses": [analysis.to_dict() for analysis in failure_analyses],
                "reconstructions": [recon.to_dict() for recon in reconstructions]
            })

            # 반복 히스토리 업데이트
            _update_iteration_history(state, failed_subqueries, reconstructions)

            logger.info(f"[{agent_name}] 재계획 완료: {len(reconstructions)}개 쿼리 재구성")

            # 🔧 수정: 업데이트된 metadata를 명시적으로 전달
            updated_metadata = state.get("metadata", {})

            return type(state)(
                messages=state.get("messages", []),
                context_messages=state.get("context_messages", []),
                metadata=updated_metadata,  # ← 업데이트된 metadata 사용
                current_agent=agent_name,
                session_id=state.get("session_id", "")
            )

        except Exception as e:
            logger.error(f"[{agent_name}] 재계획 프로세스 실패: {e}")
            _update_replanner_metadata(state, {
                "status": "error",
                "error_message": str(e),
                "iteration_count": iteration_count
            })

            # 🔧 수정: 업데이트된 metadata를 명시적으로 전달
            updated_metadata = state.get("metadata", {})

            return type(state)(
                messages=state.get("messages", []),
                context_messages=state.get("context_messages", []),
                metadata=updated_metadata,  # ← 업데이트된 metadata 사용
                current_agent=agent_name,
                session_id=state.get("session_id", "")
            )

    return replanning_node_impl


def _identify_failed_subqueries(state: AgentState) -> List[str]:
    """실패한 서브쿼리 식별"""

    quality_data = MetadataManager.safe_get_metadata(state, "quality_evaluator")
    if not quality_data:
        return []

    quality_results = quality_data.get("evaluation_results", {})
    failed_subqueries = []

    for subquery_id, quality_info in quality_results.items():
        if not quality_info.get("is_relevant", False):
            failed_subqueries.append(subquery_id)

    return failed_subqueries


def _measure_keyword_change_ratio(original: str, reconstructed: str) -> float:
    """키워드 변경 비율 측정 (30% 이상 변경 요구)

    Args:
        original: 원본 쿼리
        reconstructed: 재구성된 쿼리

    Returns:
        float: 키워드 변경 비율 (0.0-1.0, 1.0 = 100% 변경)
    """
    original_keywords = set(original.lower().split())
    reconstructed_keywords = set(reconstructed.lower().split())

    if not original_keywords:
        return 1.0  # 원본이 비어있으면 완전히 변경된 것으로 간주

    # 대칭 차집합: 원본에만 있거나 재구성에만 있는 키워드
    changed_keywords = original_keywords.symmetric_difference(reconstructed_keywords)
    change_ratio = len(changed_keywords) / len(original_keywords)

    return change_ratio


def _is_too_similar(text1: str, text2: str, threshold: float = 0.80) -> bool:
    """두 텍스트가 너무 유사한지 검증 (재구성 품질 체크)

    Args:
        text1: 첫 번째 텍스트 (원본 쿼리)
        text2: 두 번째 텍스트 (재구성된 쿼리)
        threshold: 유사도 임계값 (0.80 = 80% 유사) - 실질적 변화 요구

    Returns:
        True if texts are too similar (재구성 실패), False otherwise
    """
    from difflib import SequenceMatcher

    # 원본이 비어있으면 무조건 재구성 허용
    if not text1.strip():
        return False

    # 정규화: 소문자 변환 및 공백 정리
    normalized_text1 = " ".join(text1.lower().split())
    normalized_text2 = " ".join(text2.lower().split())

    # 완전 동일한 경우
    if normalized_text1 == normalized_text2:
        return True

    # 1. 문자열 유사도 계산 (0.80 임계값)
    ratio = SequenceMatcher(None, normalized_text1, normalized_text2).ratio()

    # 2. 키워드 변경 비율 계산 (30% 이상 요구)
    keyword_change_ratio = _measure_keyword_change_ratio(text1, text2)

    # 유사도 80% 이상 OR 키워드 변경 30% 미만이면 "too similar"
    is_similar = ratio >= threshold or keyword_change_ratio < 0.30

    return is_similar


def _update_planner_with_reconstructions(
    state: AgentState,
    reconstructions: List[ReconstructionResult]
) -> None:
    """재구성된 쿼리로 플래너 데이터 업데이트 (품질 검증 포함)"""

    logger = get_logger(__name__)
    planner_data = MetadataManager.safe_get_metadata(state, "planner")
    if not planner_data:
        return

    planner_json = planner_data.get("last_plan_json", {})
    subqueries_list = planner_json.get("subqueries", [])

    # 재구성 품질 통계
    unchanged_count = 0
    total_reconstructions = len(reconstructions)

    # 재구성 결과를 딕셔너리로 변환 (빠른 lookup)
    reconstruction_dict = {r.subquery_id: r for r in reconstructions}

    # 리스트를 직접 순회하며 업데이트
    for subquery in subqueries_list:
        subquery_id = subquery.get("id")
        if subquery_id in reconstruction_dict:
            reconstruction = reconstruction_dict[subquery_id]
            old_text = subquery.get("text", "")
            new_text = reconstruction.reconstructed_query

            # 디버깅: 재구성 전후 쿼리 출력
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, old_text.lower(), new_text.lower()).ratio()
            logger.info(
                f"🔍 [{subquery_id}] Reconstruction comparison:\n"
                f"  Original: '{old_text}'\n"
                f"  Reconstructed: '{new_text}'\n"
                f"  Similarity: {similarity:.2%}"
            )

            # 재구성 품질 검증
            if old_text == new_text or _is_too_similar(old_text, new_text):
                logger.warning(
                    f"❌ [{subquery_id}] Reconstruction quality check failed - "
                    f"new query too similar to original (similarity: {similarity:.2%}). Skipping update."
                )
                unchanged_count += 1
                continue  # 품질 검증 실패 - 업데이트 skip

            # 품질 검증 통과 - 업데이트 진행
            logger.info(f"✅ [{subquery_id}] Reconstruction quality OK - updating query")
            subquery["text"] = new_text
            subquery["reconstruction_info"] = {
                "iteration_count": reconstruction.iteration_count,
                "improvement_rationale": reconstruction.improvement_rationale,
                "confidence_score": reconstruction.confidence_score,
                "original_query": reconstruction.original_query
            }

    # 재구성 품질 메타데이터 설정
    reconstruction_quality = "high"
    if unchanged_count > 0:
        quality_ratio = unchanged_count / total_reconstructions
        if quality_ratio >= 0.5:  # 50% 이상 실패
            reconstruction_quality = "low"
            logger.error(
                f"🚨 Reconstruction quality too low: {unchanged_count}/{total_reconstructions} "
                f"({quality_ratio*100:.1f}%) failed similarity check"
            )
        elif quality_ratio >= 0.3:  # 30% 이상 실패
            reconstruction_quality = "medium"
            logger.warning(
                f"⚠️ Reconstruction quality medium: {unchanged_count}/{total_reconstructions} "
                f"({quality_ratio*100:.1f}%) failed similarity check"
            )

    # 플래너 메타데이터 업데이트
    if "metadata" not in state:
        state["metadata"] = {}
    if "planner" not in state["metadata"]:
        state["metadata"]["planner"] = {}
    state["metadata"]["planner"]["last_plan_json"] = planner_json

    # 재구성 품질 정보 저장
    if "replanner" not in state["metadata"]:
        state["metadata"]["replanner"] = {}
    state["metadata"]["replanner"]["reconstruction_quality"] = reconstruction_quality
    state["metadata"]["replanner"]["reconstruction_stats"] = {
        "total": total_reconstructions,
        "successful": total_reconstructions - unchanged_count,
        "failed": unchanged_count,
        "success_rate": (total_reconstructions - unchanged_count) / total_reconstructions if total_reconstructions > 0 else 0
    }


def _update_replanner_metadata(state: AgentState, update_data: Dict[str, Any]) -> None:
    """재계획자 메타데이터 업데이트"""

    if "metadata" not in state:
        state["metadata"] = {}
    if "replanner" not in state["metadata"]:
        state["metadata"]["replanner"] = {}

    state["metadata"]["replanner"].update(update_data)


def _update_iteration_history(
    state: AgentState,
    failed_subqueries: List[str],
    reconstructions: List[ReconstructionResult]
) -> None:
    """반복 히스토리 업데이트"""

    # 현재 반복 정보 구성
    current_iteration = {
        "iteration": MetadataManager.safe_get_metadata(state, "replanner", "iteration_count") or 1,
        "failed_count": len(failed_subqueries),
        "reconstructed_count": len(reconstructions),
        "average_score": sum(r.confidence_score for r in reconstructions) / len(reconstructions) if reconstructions else 0.0,
        "timestamp": "current"  # 실제 구현시 datetime 사용
    }

    # 히스토리에 추가
    if "metadata" not in state:
        state["metadata"] = {}
    if "replanner" not in state["metadata"]:
        state["metadata"]["replanner"] = {}

    iteration_history = state["metadata"]["replanner"].get("iteration_history", [])
    iteration_history.append(current_iteration)

    # 최대 10개 히스토리 유지
    if len(iteration_history) > 10:
        iteration_history = iteration_history[-10:]

    state["metadata"]["replanner"]["iteration_history"] = iteration_history
