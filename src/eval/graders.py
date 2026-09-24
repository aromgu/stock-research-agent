"""Phase 5: 자동 채점 로직.

1) 수치 정확도: LLM 답변 텍스트에서 숫자/비율/배수를 뽑아 DART·KRX 원본값과 대조.
   numeric_target["kind"]로 채점 방식을 고른다:
   - "amount"(기본): 원 단위 금액 (DART DB 조회)
   - "ratio": 비율(%) — 두 계정과목의 비 (예: 영업이익률)
   - "ratio_pair": 두 회사의 비율(%)을 각각 따로 채점 (기업 비교 질문용)
   - "price_return": 기간 주가 수익률(%) (pykrx)
   - "per": PER 배수 (pykrx, KRX 로그인 필요 — 로그인 정보 없으면 채점 생략)
   - "price_direction": 주가 방향(상승/하락) 일치 여부 — 잘못된 전제 질문 채점용
2) 인용 정확도: 질문별 "정답 출처" 라벨 vs 실제 사용된 도구로 Precision/Recall/F1 집계.
"""

import re
import sqlite3
from pathlib import Path

from ..data import dart_client as dc
from ..data import price_client as pc

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "financials.db"

ALL_SOURCE_LABELS = {"dart", "news", "price"}

_UNIT = {"조": 10**12, "억": 10**8, "만": 10**4}
_AMOUNT_TOLERANCE = 0.01  # LLM이 "약" 붙여 반올림해 말할 수 있어 1% 오차까지 정답으로 인정
_RATIO_TOLERANCE_PP = 0.5  # 비율(%)은 상대오차 대신 %p(퍼센트포인트) 절대오차로 비교
_PER_TOLERANCE_PP = 0.5


def _get_amount(company: str, bsns_year: str, reprt_code: str, account_name: str, amount_field: str) -> int | None:
    if amount_field not in ("thstrm_amount", "thstrm_add_amount"):
        raise ValueError(f"알 수 없는 amount_field: {amount_field}")
    corp_code = dc.get_corp_code(company)
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            f"""
            SELECT {amount_field} FROM financials
            WHERE corp_code=? AND bsns_year=? AND reprt_code=? AND account_name=? AND fs_div='CFS'
            LIMIT 1
            """,
            (corp_code, bsns_year, reprt_code, account_name),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_ground_truth(numeric_target: dict) -> int | None:
    """kind="amount"(기본) 좌표로 DART DB에서 정답 금액을 조회. 다른 kind는 채점 함수가 각자 계산."""
    amount_field = numeric_target.get("amount_field", "thstrm_amount")
    return _get_amount(
        numeric_target["company"],
        numeric_target["bsns_year"],
        numeric_target["reprt_code"],
        numeric_target["account_name"],
        amount_field,
    )


def _ratio_ground_truth(nt: dict) -> float | None:
    amount_field = nt.get("amount_field", "thstrm_add_amount")
    num = _get_amount(nt["company"], nt["bsns_year"], nt["reprt_code"], nt["numerator_account"], amount_field)
    den = _get_amount(nt["company"], nt["bsns_year"], nt["reprt_code"], nt["denominator_account"], amount_field)
    if num is None or den is None or den == 0:
        return None
    return num / den * 100


def _price_return_ground_truth(nt: dict) -> float | None:
    ticker = dc.get_stock_code(nt["company"])
    rows = pc.get_ohlcv(ticker, nt["start"], nt["end"])
    if not rows:
        return None
    return (rows[-1]["close"] / rows[0]["close"] - 1) * 100


def _per_ground_truth(nt: dict) -> float | None:
    ticker = dc.get_stock_code(nt["company"])
    fund = pc.get_fundamental(ticker, nt["on"])
    return fund["PER"] if fund else None


def _parse_composite(text: str) -> int | None:
    """'74조 6,600억' 같은 조/억/만 조합 표현을 원 단위 정수로 변환."""
    total = 0.0
    found = False
    for num, unit in re.findall(r"([\d,]+(?:\.\d+)?)\s*(조|억|만)", text):
        total += float(num.replace(",", "")) * _UNIT[unit]
        found = True
    return int(total) if found else None


def extract_candidate_numbers(text: str) -> list[int]:
    """답변 텍스트에서 원 단위로 환산 가능한 숫자 후보를 전부 뽑는다."""
    candidates = []
    for m in re.finditer(r"(?:[\d,]+(?:\.\d+)?\s*(?:조|억|만)\s*)+", text):
        val = _parse_composite(m.group())
        if val is not None:
            candidates.append(val)
    for m in re.finditer(r"\d{1,3}(?:,\d{3})+", text):  # 콤마 원단위 그대로 echo된 경우
        candidates.append(int(m.group().replace(",", "")))
    return candidates


def extract_percentages(text: str) -> list[float]:
    """'32.5%', '9.77퍼센트' 같은 표현에서 퍼센트 값을 뽑는다 (부호도 인식: -4.2%)."""
    return [float(m) for m in re.findall(r"([+-]?\d+(?:\.\d+)?)\s*(?:%|퍼센트|프로)", text)]


def extract_multiples(text: str) -> list[float]:
    """'43.22배' 같은 배수 표현(PER/PBR 등)에서 값을 뽑는다."""
    return [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*배", text)]


_UP_WORDS = ("올랐", "상승", "올라", "증가", "뛰었", "급등")
_DOWN_WORDS = ("내렸", "하락", "내려", "감소", "떨어졌", "떨어져", "급락", "빠졌")


def _stated_direction_matches(text: str, true_direction: str) -> bool:
    if true_direction == "up":
        return any(w in text for w in _UP_WORDS)
    if true_direction == "down":
        return any(w in text for w in _DOWN_WORDS)
    return False  # true_direction == "flat" 같은 애매한 경우는 자동 채점 대상 아님


def grade_numeric(answer: str, numeric_target: dict | None) -> dict:
    """숫자/비율/방향 채점 결과: {"applicable", "correct", "ground_truth", "candidates"}."""
    if numeric_target is None:
        return {"applicable": False, "correct": None, "ground_truth": None, "candidates": []}

    kind = numeric_target.get("kind", "amount")

    if kind == "amount":
        truth = get_ground_truth(numeric_target)
        candidates = extract_candidate_numbers(answer)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        correct = any(abs(c - truth) / max(abs(truth), 1) < _AMOUNT_TOLERANCE for c in candidates)
        return {"applicable": True, "correct": correct, "ground_truth": truth, "candidates": candidates}

    if kind == "ratio":
        truth = _ratio_ground_truth(numeric_target)
        candidates = extract_percentages(answer)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        correct = any(abs(c - truth) < _RATIO_TOLERANCE_PP for c in candidates)
        return {"applicable": True, "correct": correct, "ground_truth": round(truth, 2), "candidates": candidates}

    if kind == "ratio_pair":
        # 기업 비교 질문: "두 회사 수치를 각각 얼마나 정확히 댔는지"만 본다 — "어느 쪽이 더
        # 높다고 결론 내렸는지"까지는 텍스트에서 안정적으로 파싱하기 어려워 채점 범위에서 뺐다
        # (알려진 단순화, [[project_stock_research_agent]] 참고).
        truth_a = _ratio_ground_truth(numeric_target["company_a"])
        truth_b = _ratio_ground_truth(numeric_target["company_b"])
        candidates = extract_percentages(answer)
        if truth_a is None or truth_b is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        found_a = any(abs(c - truth_a) < _RATIO_TOLERANCE_PP for c in candidates)
        found_b = any(abs(c - truth_b) < _RATIO_TOLERANCE_PP for c in candidates)
        return {
            "applicable": True,
            "correct": found_a and found_b,
            "ground_truth": (round(truth_a, 2), round(truth_b, 2)),
            "candidates": candidates,
        }

    if kind == "price_return":
        truth = _price_return_ground_truth(numeric_target)
        candidates = extract_percentages(answer)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        correct = any(abs(c - truth) < _RATIO_TOLERANCE_PP for c in candidates)
        return {"applicable": True, "correct": correct, "ground_truth": round(truth, 2), "candidates": candidates}

    if kind == "per":
        truth = _per_ground_truth(numeric_target)  # KRX 로그인 없으면 None
        candidates = extract_multiples(answer)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        correct = any(abs(c - truth) < _PER_TOLERANCE_PP for c in candidates)
        return {"applicable": True, "correct": correct, "ground_truth": round(truth, 2), "candidates": candidates}

    if kind == "price_direction":
        truth = _price_return_ground_truth(numeric_target)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": []}
        true_direction = "up" if truth > 0 else "down" if truth < 0 else "flat"
        correct = _stated_direction_matches(answer, true_direction)
        return {"applicable": True, "correct": correct, "ground_truth": true_direction, "candidates": []}

    raise ValueError(f"알 수 없는 numeric_target kind: {kind}")


_DEFER_PHRASES = (
    "말씀해 주실래요", "말씀해주실래요", "알려주시면", "진행할까요", "확인해도 될까요",
    "필요할까요", "어떤 관점", "어느 쪽을", "다시 찾아볼", "여쭤봐도", "괜찮을까요", "궁금하신",
)  # fmt: skip


def is_deferred_answer(text: str) -> bool:
    """답변이 실질적인 결론 대신 사용자에게 되묻는 질문으로 끝나는지 휴리스틱 탐지.

    citation/numeric 채점만으로는 "숫자는 맞게 냈지만 정작 질문의 핵심(원인 설명)은
    회피하고 되물었다" 같은 경우를 못 잡는다(Phase 5 q5 사례에서 실제로 발견) — 별도
    축으로 잡아서 두 채점을 다 통과해도 이게 True면 "완결된 답변은 아니다"로 표시한다.
    완벽한 판별은 LLM-judge가 필요하지만, 여기선 값싸게 잡을 수 있는 신호만 본다.
    """
    tail = text.strip()[-200:]  # 답변 끝부분만 봄 (도입부 예시에 물음표가 섞여있을 수 있어서)
    return tail.rstrip().endswith(("?", "요?", "까요?")) and any(p in tail for p in _DEFER_PHRASES)


def accumulate_citation_counts(predicted: set[str], gold: set[str], counts: dict) -> None:
    """예측된 출처 집합 vs 정답 출처 집합으로 TP/FP/FN을 counts(dict)에 누적."""
    for label in ALL_SOURCE_LABELS:
        if label in predicted and label in gold:
            counts["tp"] += 1
        elif label in predicted and label not in gold:
            counts["fp"] += 1
        elif label not in predicted and label in gold:
            counts["fn"] += 1


def precision_recall_f1(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {"precision": precision, "recall": recall, "f1": f1}
