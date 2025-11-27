"""
Loop Prevention Module - Routing Loop Detection and Prevention
==============================================================

LORIN 시스템의 라우팅 루프 방지 시스템입니다.
무한 반복을 감지하고 강제 종료하는 안전장치를 제공합니다.

주요 기능:
- 글로벌 반복 횟수 추적
- 다양한 루프 패턴 감지 (진동, 연속, 순환)
- 강제 종료 및 로깅
- 루프 방지 메타데이터 관리
"""

from typing import Dict, List, Any
from datetime import datetime, timezone

from ...logger.logger import get_logger
from ..schema import (
    LoopPreventionData,
    LoopDetectionResult,
    PatternDetectionResult,
    RoutingDecision,
    LastRoutingDecision,
    MetadataManager,
)

logger = get_logger(__name__)

# ============================================
# 루프 방지 시스템 구현
# ============================================


def initialize_loop_prevention(metadata: Dict[str, Any]) -> LoopPreventionData:
    """루프 방지 시스템 초기화 및 기존 데이터 로드

    Args:
        metadata: AgentState 메타데이터

    Returns:
        LoopPreventionData: 루프 방지 데이터 구조
    """
    loop_data = metadata.get("loop_prevention", {})

    # 첫 실행 시 초기화
    if not loop_data:
        loop_data = {
            "global_iteration_count": 0,
            "routing_history": [],
            "last_routes": [],
            "pattern_cache": {},
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "forced_terminations": 0,
            "oscillation_count": 0
        }

    # 현재 반복 카운트 증가
    loop_data["global_iteration_count"] += 1

    return loop_data


def detect_routing_loop(loop_prevention: LoopPreventionData) -> LoopDetectionResult:
    """라우팅 루프 패턴 감지

    Args:
        loop_prevention: 루프 방지 데이터

    Returns:
        LoopDetectionResult: 루프 감지 결과
    """
    routing_history = loop_prevention.get("routing_history", [])

    # 라우팅 히스토리가 부족하면 루프 없음 (2회 연속 감지를 위해 임계값 낮춤)
    if len(routing_history) < 3:
        return {"is_loop": False, "pattern": None, "confidence": 0.0}

    # 최근 경로들 분석
    recent_routes = routing_history[-6:]  # 최근 6개 라우팅 추출

    # 패턴 1: A-B-A-B 진동 패턴 감지
    oscillation_detected = detect_oscillation_pattern(recent_routes)
    if oscillation_detected["detected"]:
        return {
            "is_loop": True,
            "pattern": f"Oscillation: {oscillation_detected['pattern']}",
            "confidence": oscillation_detected["confidence"]
        }

    # 패턴 2: 3회 연속 동일 경로 감지
    consecutive_detected = detect_consecutive_pattern(recent_routes)
    if consecutive_detected["detected"]:
        return {
            "is_loop": True,
            "pattern": f"Consecutive: {consecutive_detected['pattern']}",
            "confidence": consecutive_detected["confidence"]
        }

    # 패턴 3: 복잡한 순환 패턴 감지 (A-B-C-A-B-C)
    cycle_detected = detect_cycle_pattern(recent_routes)
    if cycle_detected["detected"]:
        return {
            "is_loop": True,
            "pattern": f"Cycle: {cycle_detected['pattern']}",
            "confidence": cycle_detected["confidence"]
        }

    return {"is_loop": False, "pattern": None, "confidence": 0.0}


def detect_oscillation_pattern(routes: List[str]) -> PatternDetectionResult:
    """A-B-A-B 진동 패턴 감지

    Args:
        routes: 라우팅 히스토리 목록

    Returns:
        PatternDetectionResult: 감지 결과
    """
    if len(routes) < 4:
        return {"detected": False, "pattern": None, "confidence": 0.0}

    # 최근 4개 라우팅에서 A-B-A-B 패턴 찾기
    for i in range(len(routes) - 3):
        pattern_slice = routes[i:i+4]
        if (pattern_slice[0] == pattern_slice[2] and
            pattern_slice[1] == pattern_slice[3] and
            pattern_slice[0] != pattern_slice[1]):

            pattern_str = f"{pattern_slice[0]}-{pattern_slice[1]}"
            confidence = 0.9  # 높은 신뢰도

            return {
                "detected": True,
                "pattern": pattern_str,
                "confidence": confidence
            }

    return {"detected": False, "pattern": None, "confidence": 0.0}


def detect_consecutive_pattern(routes: List[str]) -> PatternDetectionResult:
    """연속 동일 라우팅 패턴 감지 (25회 연속으로 완화)

    Args:
        routes: 라우팅 히스토리 목록

    Returns:
        PatternDetectionResult: 감지 결과
    """
    if len(routes) < 25:
        return {"detected": False, "pattern": None, "confidence": 0.0}

    # 최근 25개가 모두 replanner인지 확인 (replanner 루프 방지 - 완화됨)
    recent_25 = routes[-25:]
    if len(recent_25) == 25 and all(route == "replanner" for route in recent_25):
        pattern_str = "25x-replanner"
        confidence = 0.95  # 매우 높은 신뢰도 - 25회 연속은 확실한 루프
        return {"detected": True, "pattern": pattern_str, "confidence": confidence}

    # 최근 25개가 모두 retrieve인지 확인 (retrieve 루프 방지)
    if len(recent_25) == 25 and all(route == "retrieve" for route in recent_25):
        pattern_str = "25x-retrieve"
        confidence = 0.9
        return {"detected": True, "pattern": pattern_str, "confidence": confidence}

    # 최근 25개가 모두 동일한지 확인 (완화됨)
    if len(routes) >= 25:
        recent_25_for_general = routes[-25:]
        if len(set(recent_25_for_general)) == 1:  # 모두 동일
            pattern_str = f"25x-{recent_25_for_general[0]}"
            confidence = 0.8
            return {
                "detected": True,
                "pattern": pattern_str,
                "confidence": confidence
            }

    return {"detected": False, "pattern": None, "confidence": 0.0}


def detect_cycle_pattern(routes: List[str]) -> PatternDetectionResult:
    """복잡한 순환 패턴 감지 (A-B-C-A-B-C)

    Cycle 감지 비활성화: "성공할 때까지 반복" 구조 허용
    - replanner 연속 패턴은 정상적인 순환 구조
    - 실제 플로우: replanner -> retrieve -> quality_evaluator -> replanner -> ...

    Args:
        routes: 라우팅 히스토리 목록

    Returns:
        PatternDetectionResult: 감지 결과 (항상 False 반환)
    """
    # Cycle 감지 완전 비활성화
    return {"detected": False, "pattern": None, "confidence": 0.0}


def log_forced_termination(
    reason: str,
    loop_prevention: LoopPreventionData,
    iteration: int
) -> None:
    """강제 종료 로깅

    Args:
        reason: 종료 사유
        loop_prevention: 루프 방지 데이터
        iteration: 현재 반복 횟수
    """
    loop_prevention["forced_terminations"] += 1

    logger.warning(f"[FORCED TERMINATION] {reason}")
    logger.warning(f"[FORCED TERMINATION] Global iteration: {iteration}")
    logger.warning(f"[FORCED TERMINATION] Routing history: {loop_prevention.get('routing_history', [])[-10:]}")
    logger.warning(f"[FORCED TERMINATION] Total forced terminations: {loop_prevention['forced_terminations']}")


def update_loop_prevention_metadata(
    state: Dict[str, Any],
    route: str,
    routing_decision: RoutingDecision,
    loop_prevention: LoopPreventionData
) -> None:
    """루프 방지 메타데이터 업데이트

    Args:
        state: AgentState (직접 수정)
        route: 선택된 라우팅
        routing_decision: 라우팅 결정 정보
        loop_prevention: 루프 방지 데이터
    """
    # 라우팅 히스토리 업데이트
    loop_prevention["routing_history"].append(route)

    # 히스토리 크기 제한 (최근 50개만 유지)
    if len(loop_prevention["routing_history"]) > 50:
        loop_prevention["routing_history"] = loop_prevention["routing_history"][-50:]

    # MetadataManager를 통한 안전한 메타데이터 저장
    # loop_prevention 데이터 저장
    MetadataManager.safe_set_metadata(
        state,
        "loop_prevention",
        loop_prevention,
        add_timestamp=False  # loop_prevention에는 자체 타임스탬프가 있음
    )

    # 라우팅 결정 정보 저장
    last_routing_decision: LastRoutingDecision = {
        "route": route,
        "reason": routing_decision.get("reason", ""),
        "strategy": routing_decision.get("strategy", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_iteration": loop_prevention["global_iteration_count"]
    }
    MetadataManager.safe_set_metadata(
        state,
        "last_routing_decision",
        last_routing_decision,
        add_timestamp=False  # 이미 timestamp 필드가 있음
    )


# ============================================
# Backward Compatibility - 기존 private 함수명 유지
# ============================================

# 기존 코드와의 호환성을 위해 private 함수명으로 alias 제공
_initialize_loop_prevention = initialize_loop_prevention
_detect_routing_loop = detect_routing_loop
_detect_oscillation_pattern = detect_oscillation_pattern
_detect_consecutive_pattern = detect_consecutive_pattern
_detect_cycle_pattern = detect_cycle_pattern
_log_forced_termination = log_forced_termination
_update_loop_prevention_metadata = update_loop_prevention_metadata
