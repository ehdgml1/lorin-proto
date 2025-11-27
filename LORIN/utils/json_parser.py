"""
JSON Parser Utility - 통합된 JSON 파싱 로직
==========================================

planner, replanner, quality_evaluator에서 중복 사용되는 JSON 파싱 로직을 통합
"""

import json
import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """LLM 응답에서 JSON 추출 - 통합된 안정적인 방식

    Args:
        response: LLM 응답 문자열

    Returns:
        Optional[Dict[str, Any]]: 파싱된 JSON 또는 None

    Features:
        - 코드 블록 감지 (```json, ```)
        - 균형잡힌 중괄호 검색
        - 주석 제거
        - 후행 쉼표 처리
    """
    if not response:
        return None

    # 0. 순수 JSON 체크: 이미 올바른 형태라면 전처리 건너뛰기
    stripped = response.strip()
    if stripped.startswith('{') and stripped.endswith('}') and '```' not in stripped[:50]:
        # 순수 JSON으로 보이면 전처리 없이 직접 파싱 시도
        logger.info(f"[json_parser] Pure JSON detected ({len(stripped)} chars), skipping preprocessing")
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                # 우선순위 키 체크
                if 'subqueries' in parsed or 'initial_query' in parsed or 'expansion_queries' in parsed:
                    logger.info(f"[json_parser] ✓ Successfully parsed pure JSON with priority keys: {list(parsed.keys())}")
                    return parsed
        except json.JSONDecodeError as e:
            logger.info(f"[json_parser] Pure JSON parsing failed: {e}, falling back to preprocessing")

    # 0. 사전 정리 (순수 JSON 파싱 실패 시에만)
    response = _preprocess_llm_response(response)

    # 1. 코드 블록 패턴 시도
    json_code_patterns = [
        r'```json\s*(\{[\s\S]*?\})\s*```',      # 표준 json 코드블록
        r'```\s*(\{[\s\S]*?\})\s*```',           # 언어 태그 없는 코드블록
        r'```json\s*(\[[\s\S]*?\])\s*```',       # JSON 배열
        r'```JSON\s*(\{[\s\S]*?\})\s*```',       # 대문자 JSON
    ]

    for pattern in json_code_patterns:
        matches = re.finditer(pattern, response, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            json_candidate = match.group(1) if match.groups() else match.group(0)

            # 점진적 정리 및 파싱 시도
            for cleaned in _progressive_json_cleaning(json_candidate):
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, (dict, list)):
                        return parsed if isinstance(parsed, dict) else {"data": parsed}
                except json.JSONDecodeError:
                    continue

    # 2. 균형잡힌 중괄호 블록 찾기
    json_candidates = _find_balanced_json_blocks(response)
    logger.info(f"[json_parser] Found {len(json_candidates)} JSON candidates")
    for i, cand in enumerate(json_candidates):
        logger.info(f"[json_parser] Candidate {i}: {len(cand)} chars, preview: {cand[:100]}...")

    # 우선순위: 'subqueries' > 'initial_query' > 'expansion_queries' > 가장 큰 객체
    prioritized_candidates = []
    for idx, candidate in enumerate(json_candidates):
        logger.info(f"[json_parser] Processing candidate {idx} ({len(candidate)} chars)")
        for cleaned in _progressive_json_cleaning(candidate):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    keys = list(parsed.keys())
                    logger.info(f"[json_parser] Candidate {idx} parsed successfully, keys: {keys}")
                    # 'subqueries' 키가 있으면 최우선으로 반환
                    if 'subqueries' in parsed:
                        logger.info(f"[json_parser] ✓ Returning candidate {idx} with 'subqueries' key ({len(candidate)} chars)")
                        return parsed
                    # 'initial_query' 키가 있으면 우선 반환 (sequential mode)
                    if 'initial_query' in parsed:
                        logger.info(f"[json_parser] ✓ Returning candidate {idx} with 'initial_query' key ({len(candidate)} chars)")
                        return parsed
                    # 'expansion_queries' 키가 있으면 우선 반환 (sequential mode)
                    if 'expansion_queries' in parsed:
                        logger.info(f"[json_parser] ✓ Returning candidate {idx} with 'expansion_queries' key ({len(candidate)} chars)")
                        return parsed
                    # 없으면 나중을 위해 저장
                    prioritized_candidates.append((len(candidate), parsed))
                    logger.info(f"[json_parser] Candidate {idx} added to prioritized list (no priority keys)")
                    break  # ← 첫 번째 성공한 cleaned 버전만 사용
            except json.JSONDecodeError as e:
                logger.debug(f"[json_parser] Candidate {idx} parsing failed: {e}")
                continue

    # 특정 키 없으면 가장 큰 유효한 JSON 반환
    if prioritized_candidates:
        prioritized_candidates.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"[json_parser] Returning largest valid JSON ({prioritized_candidates[0][0]} chars)")
        return prioritized_candidates[0][1]

    # 3. 첫 { 부터 마지막 } 까지 추출 (최후 수단)
    try:
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            blob = response[start : end + 1]
            parsed = json.loads(blob)
            # 'subqueries' 키 확인
            if isinstance(parsed, dict) and 'subqueries' in parsed:
                logger.debug(f"[json_parser] Fallback: Found JSON with 'subqueries' key")
                return parsed
            return parsed
    except json.JSONDecodeError:
        pass

    # 4. 최후의 수단: 불완전한 JSON 복구 시도 (응답이 잘린 경우)
    try:
        start = response.find('{"subqueries"')
        if start != -1:
            # "subqueries"로 시작하는 JSON을 찾았으니 복구 시도
            end = response.rfind("}")
            if end > start:
                incomplete_json = response[start:end + 1]
                # 닫히지 않은 배열이나 객체를 닫아주기
                open_braces = incomplete_json.count("{") - incomplete_json.count("}")
                open_brackets = incomplete_json.count("[") - incomplete_json.count("]")

                # 복구 시도
                repaired = incomplete_json
                for _ in range(open_brackets):
                    repaired += "]"
                for _ in range(open_braces):
                    repaired += "}"

                parsed = json.loads(repaired)
                if isinstance(parsed, dict) and 'subqueries' in parsed:
                    logger.info(f"[json_parser] Repaired incomplete JSON (added {open_brackets} ']' and {open_braces} '}}')")
                    return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def sanitize_json_string(json_str: str) -> str:
    """JSON 문자열 정제 - 제어 문자, escape, 주석 처리

    Args:
        json_str: 원본 JSON 문자열

    Returns:
        str: 정제된 JSON 문자열
    """
    if not json_str:
        return ""

    # 1. 주석 제거
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    # 2. 문자열 값 내의 제어 문자 처리
    def fix_string_content(match):
        string_content = match.group(1)

        control_char_map = {
            '\n': '\\n',
            '\r': '\\r',
            '\t': '\\t',
            '\b': '\\b',
            '\f': '\\f',
        }

        result = []
        i = 0
        while i < len(string_content):
            char = string_content[i]

            # 이미 올바르게 escape된 문자는 유지
            if char == '\\' and i + 1 < len(string_content):
                next_char = string_content[i + 1]
                if next_char in '"\\bfnrt/u':
                    result.append(char)
                    result.append(next_char)
                    i += 2
                    continue
                else:
                    result.append('\\\\')
                    i += 1
                    continue

            # 제어 문자를 escape 시퀀스로 변환
            if char in control_char_map:
                result.append(control_char_map[char])
            elif ord(char) < 0x20 or (0x7F <= ord(char) <= 0x9F):
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)

            i += 1

        return f'"{"".join(result)}"'

    # JSON 문자열 값들을 찾아서 처리
    json_str = re.sub(r'"((?:[^"\\]|\\.)*)?"', fix_string_content, json_str)

    return json_str


# ─────────────────────────────────────────────────────────
# Private Helper Functions
# ─────────────────────────────────────────────────────────

def _preprocess_llm_response(response: str) -> str:
    """LLM 응답 사전 정리"""
    # JSON 이전 텍스트 제거
    response = re.sub(r'^.*?(?=\{)', '', response, flags=re.DOTALL)

    # 마지막 } 이후 텍스트만 제거
    last_brace_pos = response.rfind('}')
    if last_brace_pos != -1:
        response = response[:last_brace_pos + 1]

    # 마크다운 아티팩트 제거
    response = re.sub(r'^```[a-zA-Z]*\n?', '', response, flags=re.MULTILINE)
    response = re.sub(r'\n?```$', '', response, flags=re.MULTILINE)

    return response.strip()


def _progressive_json_cleaning(json_str: str) -> list:
    """점진적 JSON 정리 - 여러 단계의 정리 방법 시도"""
    if not json_str:
        return []

    cleaned_versions = []

    # 1. 원본
    cleaned_versions.append(json_str.strip())

    # 2. 기본 정리
    basic_clean = json_str.strip()
    basic_clean = re.sub(r'^```[a-zA-Z]*\n?', '', basic_clean)
    basic_clean = re.sub(r'\n?```$', '', basic_clean)
    cleaned_versions.append(basic_clean)

    # 3. 주석 제거
    no_comments = re.sub(r'//.*?$', '', basic_clean, flags=re.MULTILINE)
    no_comments = re.sub(r'/\*.*?\*/', '', no_comments, flags=re.DOTALL)
    cleaned_versions.append(no_comments)

    # 4. 따옴표 정규화
    quote_fixed = no_comments.replace('"', '"').replace('"', '"')
    quote_fixed = quote_fixed.replace(''', "'").replace(''', "'")
    cleaned_versions.append(quote_fixed)

    # 5. 후행 쉼표 제거
    comma_fixed = re.sub(r',(\s*[}\]])', r'\1', quote_fixed)
    cleaned_versions.append(comma_fixed)

    # 6. 추가 공백 정리
    space_clean = re.sub(r'\s+', ' ', comma_fixed)
    space_clean = re.sub(r'\s*([{}:,\[\]])\s*', r'\1', space_clean)
    cleaned_versions.append(space_clean)

    # 중복 제거
    return list(dict.fromkeys(v for v in cleaned_versions if v))


def _find_balanced_json_blocks(text: str) -> list:
    """텍스트에서 균형잡힌 JSON 블록들 찾기"""
    candidates = []

    logger.info(f"[_find_balanced_json_blocks] Input text length: {len(text)} chars")
    logger.info(f"[_find_balanced_json_blocks] Input preview: {text[:200]}...")

    # 모든 { 위치 찾기
    brace_positions = [match.start() for match in re.finditer(r'\{', text)]
    logger.info(f"[_find_balanced_json_blocks] Found {len(brace_positions)} '{{' positions: {brace_positions[:10]}")

    for idx, match in enumerate(re.finditer(r'\{', text)):
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
                        logger.info(f"[_find_balanced_json_blocks] Brace {idx} at pos {start_pos}: Found balanced block, {len(candidate)} chars")
                        logger.info(f"[_find_balanced_json_blocks] Brace {idx} preview: {candidate[:100]}...")
                        if len(candidate) > 10:
                            candidates.append(candidate)
                            logger.info(f"[_find_balanced_json_blocks] Brace {idx}: Added to candidates")
                        else:
                            logger.info(f"[_find_balanced_json_blocks] Brace {idx}: Too short, skipped")
                        break

    logger.info(f"[_find_balanced_json_blocks] Total candidates found: {len(candidates)}")
    return candidates
