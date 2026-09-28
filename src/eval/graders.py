"""Phase 5: 자동 채점 로직.

1) 수치 정확도: LLM 답변 텍스트에서 숫자/비율/배수를 뽑아 DART·KRX 원본값과 대조.
   numeric_target["kind"]로 채점 방식을 고른다:
   - "amount"(기본): 원 단위 금액 (DART DB 조회)
   - "ratio": 비율(%) — 두 계정과목의 비 (예: 영업이익률)
   - "ratio_pair": 두 회사의 비율(%)을 각각 따로 채점 (기업 비교 질문용)
   - "growth": 전년 동기 대비 증감률(%) (DART 당기 vs 전년 동기)
   - "price_return": 기간 주가 수익률(%) (pykrx)
   - "per": PER 배수, field로 PBR 등 선택 (pykrx, KRX 로그인 필요 — 로그인 정보 없으면 채점 생략)
   - "price_direction": 주가 방향(상승/하락) 일치 여부 — 잘못된 전제 질문 채점용
2) 인용 정확도: 질문별 "정답 출처" 라벨 vs 실제 사용된 도구로 Precision/Recall/F1 집계.
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from openai import OpenAI

from ..data import dart_client as dc
from ..data import price_client as pc

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "financials.db"

ALL_SOURCE_LABELS = {"dart", "news", "price"}

_UNIT = {"조": 10**12, "억": 10**8, "만": 10**4}
_AMOUNT_TOLERANCE = 0.01  # LLM이 "약" 붙여 반올림해 말할 수 있어 1% 오차까지 정답으로 인정
_RATIO_TOLERANCE_PP = 0.5  # 비율(%)은 상대오차 대신 %p(퍼센트포인트) 절대오차로 비교
_PER_TOLERANCE_PP = 0.5


def _get_amount(company: str, bsns_year: str, reprt_code: str, account_name: str, amount_field: str) -> int | None:
    if amount_field not in ("thstrm_amount", "thstrm_add_amount", "frmtrm_amount", "frmtrm_add_amount"):
        raise ValueError(f"알 수 없는 amount_field: {amount_field}")
    corp_code = dc.get_corp_code(company)
    conn = sqlite3.connect(DB_PATH)
    try:
        # 도구(_fetch_financials)와 같은 규칙: 연결재무제표(CFS)가 없는 회사는 별도재무제표(OFS).
        # 예전엔 CFS만 봐서 자회사 없는 회사는 정답이 None이 되어 채점에서 조용히 빠졌다.
        for fs_div in ("CFS", "OFS"):
            row = conn.execute(
                f"""
                SELECT {amount_field} FROM financials
                WHERE corp_code=? AND bsns_year=? AND reprt_code=? AND account_name=? AND fs_div=?
                LIMIT 1
                """,
                (corp_code, bsns_year, reprt_code, account_name, fs_div),
            ).fetchone()
            if row:
                return row[0]
        return None
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


def _growth_amounts(nt: dict) -> tuple[int | None, int | None]:
    """(당기, 전년 동기) 금액. 기본은 누적끼리 비교 (반기보고서의 상반기 누적 vs 전년 상반기 누적)."""
    args = (nt["company"], nt["bsns_year"], nt["reprt_code"], nt["account_name"])
    return (
        _get_amount(*args, nt.get("current_field", "thstrm_add_amount")),
        _get_amount(*args, nt.get("prior_field", "frmtrm_add_amount")),
    )


def _growth_ground_truth(nt: dict) -> float | None:
    current, prior = _growth_amounts(nt)
    if current is None or not prior:
        return None
    return (current / prior - 1) * 100


def _price_return_ground_truth(nt: dict) -> float | None:
    ticker = dc.get_stock_code(nt["company"])
    rows = pc.get_ohlcv(ticker, nt["start"], nt["end"])
    if not rows:
        return None
    return (rows[-1]["close"] / rows[0]["close"] - 1) * 100


def _per_ground_truth(nt: dict) -> float | None:
    """kind="per"는 이름과 달리 field로 PER/PBR 등 KRX 밸류에이션 지표를 고를 수 있다 (기본 PER)."""
    ticker = dc.get_stock_code(nt["company"])
    fund = pc.get_fundamental(ticker, nt["on"])
    return fund[nt.get("field", "PER")] if fund else None


# 업종 1위(peer_top) 채점용 지표 계산 - tools._peer_metric과 "같은 결과가 나와야 하지만 코드는 독립"이 핵심.
# 예전엔 tools._peer_metric을 그대로 재사용해서, 그 함수에 버그가 있으면 에이전트 답과 정답이 "똑같이 틀려서"
# 통과해버리는 순환 채점이었다 (업종 이름 매칭 버그 사례처럼). financials.db는 같이 읽지만 계산 로직은 새로 짰다.
_PEER_METRIC_SPEC = {
    "영업이익률": ("영업이익", "매출액", "ratio"),
    "순이익률": ("당기순이익(손실)", "매출액", "ratio"),
    "부채비율": ("부채총계", "자본총계", "ratio"),
    "매출액 증가율": ("매출액", None, "growth"),
    "영업이익 증가율": ("영업이익", None, "growth"),
    "매출액": ("매출액", None, "amount"),
    "영업이익": ("영업이익", None, "amount"),
}
_PEER_PERIOD_RANK = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}  # tools.py와 별도로 유지 (의도적 중복)


def _peer_amount(corp_code: str, account_name: str) -> tuple[int | None, int | None, str | None]:
    """회사의 가장 최근 보고서에서 (당기 누적값, 전년 동기 누적값, 기간 라벨). 값이 없으면 (None, None, label)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        periods = conn.execute(
            "SELECT DISTINCT bsns_year, reprt_code FROM financials WHERE corp_code=?", (corp_code,)
        ).fetchall()
        if not periods:
            return None, None, None
        year, reprt_code = max(periods, key=lambda p: (int(p[0]), _PEER_PERIOD_RANK[p[1]]))
        label = f"{year}년 {reprt_code}"
        for fs_div in ("CFS", "OFS"):
            row = conn.execute(
                "SELECT thstrm_amount, thstrm_add_amount, frmtrm_amount, frmtrm_add_amount FROM financials "
                "WHERE corp_code=? AND bsns_year=? AND reprt_code=? AND account_name=? AND fs_div=? LIMIT 1",
                (corp_code, year, reprt_code, account_name, fs_div),
            ).fetchone()
            if row:
                thstrm, add, frmtrm, frmtrm_add = row
                current = add if add is not None else thstrm
                prior = frmtrm_add if frmtrm_add is not None else frmtrm
                return current, prior, label
        return None, None, label
    finally:
        conn.close()


def _peer_metric_independent(corp_code: str, metric: str) -> tuple[float | None, str | None]:
    num_name, den_name, kind = _PEER_METRIC_SPEC[metric]
    num_cur, num_prior, label = _peer_amount(corp_code, num_name)
    if num_cur is None:
        return None, label
    if kind == "amount":
        return num_cur, label
    if kind == "growth":
        return ((num_cur / num_prior - 1) * 100 if num_prior else None), label
    den_cur, _, _ = _peer_amount(corp_code, den_name)
    return ((num_cur / den_cur * 100) if den_cur else None), label


def _peer_top_ground_truth(nt: dict) -> tuple[str, float] | None:
    """업종 1위 (회사명, 지표 값). 업종 소속 회사 목록만 tools.py를 쓰고(손으로 정한 정적 데이터라 위험 낮음),
    실제 지표 계산은 위 _peer_metric_independent로 완전히 독립된 코드를 쓴다."""
    from ..agent import tools  # 채점기 로드 시 에이전트 도구 전체를 불러오지 않도록 필요할 때만 import

    _, members, _ = tools._resolve_peer_group(nt["group"])
    scored = [(c["name"], _peer_metric_independent(c["corp_code"], nt["metric"])[0]) for c in members]
    scored = [s for s in scored if s[1] is not None]
    return max(scored, key=lambda s: s[1]) if scored else None


def _parse_composite(text: str) -> int | None:
    """'74조 6,600억' 같은 조/억/만 조합 표현을 원 단위 정수로 변환."""
    total = 0.0
    found = False
    # 숫자는 반드시 숫자로 시작해야 함 — [\d,]+ 였을 때 "비용, 억제"의 ", 억"을 금액으로 읽다가 죽었다
    for num, unit in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억|만)", text):
        total += float(num.replace(",", "")) * _UNIT[unit]
        found = True
    return int(total) if found else None


def extract_candidate_numbers(text: str) -> list[int]:
    """답변 텍스트에서 원 단위로 환산 가능한 숫자 후보를 전부 뽑는다."""
    candidates = []
    for m in re.finditer(r"(?:\d[\d,]*(?:\.\d+)?\s*(?:조|억|만)\s*)+", text):
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

# "매출은 늘었는데 주가가 떨어졌다"는 전제는 맞지 않습니다 처럼, 질문의 원래 전제를 그대로 인용부호로
# 옮겨 적은 뒤 부정하는 문장이 실제 답변에 자주 나온다. 인용된 구간 안의 단어("떨어졌")까지 "모델이
# 주장한 방향"으로 잘못 집계되던 버그를 v5 divergence 문항에서 실제로 발견해 고쳤다 (2026-09-29).
# 같은 함수를 쓰는 기존 price_direction 채점(잘못된 전제 문항)에도 똑같이 적용된다.
_QUOTED_RE = re.compile(r"[“\"][^”\"]*[”\"]")

# "하락", "상승", "증가", "감소"는 명사+"하다" 어간이라 "하락하지 않았다"처럼 뒤에 부정을 붙여 그대로
# 부정문을 만들 수 있다 (반대로 "올랐"·"떨어졌" 같은 과거형 어미는 이 방식으로 부정되지 않는다).
# 매칭된 단어 바로 뒤 몇 글자 안에 "-지 않" 패턴이 있으면 그 자리의 매칭은 무시한다.
_NEGATION_WINDOW = 6
_NEGATION_MARKERS = ("지 않", "지는 않", "치 않")


def _has_unnegated_word(text: str, words: tuple[str, ...]) -> bool:
    for w in words:
        start = 0
        while (idx := text.find(w, start)) != -1:
            tail = text[idx + len(w) : idx + len(w) + _NEGATION_WINDOW]
            if not any(m in tail for m in _NEGATION_MARKERS):
                return True
            start = idx + len(w)  # 이 자리는 부정문이니 건너뛰고, 같은 단어의 다른 위치를 계속 찾는다
    return False


def _stated_direction_matches(text: str, true_direction: str) -> bool:
    text = _QUOTED_RE.sub("", text)  # 인용부호 안의 내용(대개 원래 전제를 재인용한 것)은 판정에서 제외
    if true_direction == "up":
        return _has_unnegated_word(text, _UP_WORDS)
    if true_direction == "down":
        return _has_unnegated_word(text, _DOWN_WORDS)
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

    if kind == "growth":
        truth = _growth_ground_truth(numeric_target)
        candidates = extract_percentages(answer)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        # 증가율은 수백 %까지 커서 고정 0.5%p는 너무 빡빡함 → 참값의 1%와 0.5%p 중 큰 쪽 허용
        tolerance = max(_RATIO_TOLERANCE_PP, abs(truth) * 0.01)
        pct_ok = any(abs(c - truth) < tolerance for c in candidates)
        # 증감을 %가 아니라 "전년 X원 → 당기 Y원" 금액 두 개로 제시해도 정답 (h5처럼 채점 기준이 금액을
        # 요구하는 질문에서 %만 찾으면 모든 방식이 틀린 것으로 나오던 채점 설계 오류 수정)
        current, prior = _growth_amounts(numeric_target)
        amounts = extract_candidate_numbers(answer)

        def _has(v: int) -> bool:
            return any(abs(a - v) / max(abs(v), 1) < _AMOUNT_TOLERANCE for a in amounts)

        amounts_ok = _has(current) and _has(prior)
        return {
            "applicable": True,
            "correct": pct_ok or amounts_ok,
            "ground_truth": round(truth, 2),
            "candidates": candidates + amounts,
        }

    if kind == "per":
        truth = _per_ground_truth(numeric_target)  # KRX 로그인 없으면 None
        candidates = extract_multiples(answer)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        correct = any(abs(c - truth) < _PER_TOLERANCE_PP for c in candidates)
        return {"applicable": True, "correct": correct, "ground_truth": round(truth, 2), "candidates": candidates}

    if kind == "peer_top":
        # 업종 1위 질문: 1위 회사 이름이 답변에 있고, 그 회사의 지표 값(%)도 맞아야 정답
        truth = _peer_top_ground_truth(numeric_target)
        candidates = extract_percentages(answer)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        name, value = truth
        correct = name in answer and any(abs(c - value) < _RATIO_TOLERANCE_PP for c in candidates)
        return {"applicable": True, "correct": correct, "ground_truth": (name, round(value, 2)), "candidates": candidates}

    if kind == "price_direction":
        truth = _price_return_ground_truth(numeric_target)
        if truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": []}
        true_direction = "up" if truth > 0 else "down" if truth < 0 else "flat"
        correct = _stated_direction_matches(answer, true_direction)
        return {"applicable": True, "correct": correct, "ground_truth": true_direction, "candidates": []}

    if kind == "divergence":
        # "실적은 늘었는데 주가는 왜 떨어졌어?" 유형: 실적 증감(growth)과 주가 방향(price_direction)을
        # 둘 다 맞게 짚어야 정답. numeric_target = {"kind": "divergence", "growth": {...growth 좌표...},
        # "price": {"company", "start", "end"}}
        growth_truth = _growth_ground_truth(numeric_target["growth"])
        price_truth = _price_return_ground_truth(numeric_target["price"])
        candidates = extract_percentages(answer)
        if growth_truth is None or price_truth is None:
            return {"applicable": True, "correct": None, "ground_truth": None, "candidates": candidates}
        growth_tolerance = max(_RATIO_TOLERANCE_PP, abs(growth_truth) * 0.01)
        growth_pct_ok = any(abs(c - growth_truth) < growth_tolerance for c in candidates)
        current, prior = _growth_amounts(numeric_target["growth"])
        amounts = extract_candidate_numbers(answer)

        def _has(v: int) -> bool:
            return any(abs(a - v) / max(abs(v), 1) < _AMOUNT_TOLERANCE for a in amounts)

        growth_amounts_ok = _has(current) and _has(prior)
        price_direction = "up" if price_truth > 0 else "down" if price_truth < 0 else "flat"
        price_ok = _stated_direction_matches(answer, price_direction)
        correct = (growth_pct_ok or growth_amounts_ok) and price_ok
        return {
            "applicable": True,
            "correct": correct,
            "ground_truth": {"growth_pct": round(growth_truth, 2), "price_return_pct": round(price_truth, 2), "price_direction": price_direction},
            "candidates": candidates + amounts,
        }

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


def describe_reference(numeric_target: dict | None) -> str | None:
    """심판(LLM judge)에게 줄 '참고 정답'을 DB/KRX에서 계산해 사람이 읽는 문장으로 만든다."""
    if numeric_target is None:
        return None
    kind = numeric_target.get("kind", "amount")
    nt = numeric_target
    if kind == "amount":
        truth = get_ground_truth(nt)
        if truth is None:
            return None
        return f"{nt['company']} {nt['account_name']} = {truth:,}원 (약 {truth / 10**12:.1f}조원)"
    if kind == "ratio":
        truth = _ratio_ground_truth(nt)
        if truth is None:
            return None
        return f"{nt['company']} {nt['numerator_account']}/{nt['denominator_account']} = {truth:.2f}%"
    if kind == "ratio_pair":
        a, b = nt["company_a"], nt["company_b"]
        ta, tb = _ratio_ground_truth(a), _ratio_ground_truth(b)
        if ta is None or tb is None:
            return None
        higher, lower = (a["company"], b["company"]) if ta > tb else (b["company"], a["company"])
        return f"{a['company']} {ta:.2f}%, {b['company']} {tb:.2f}% → 더 높은 쪽: {higher}, 더 낮은 쪽: {lower}"
    if kind in ("price_return", "price_direction"):
        truth = _price_return_ground_truth(nt)
        if truth is None:
            return None
        direction = "상승" if truth > 0 else "하락" if truth < 0 else "보합"
        return f"{nt['company']} 주가 {nt['start']}~{nt['end']} 수익률 {truth:+.2f}% (실제 방향: {direction})"
    if kind == "growth":
        current, prior = _growth_amounts(nt)
        truth = _growth_ground_truth(nt)
        if truth is None:
            return None
        direction = "증가" if truth > 0 else "감소" if truth < 0 else "변화 없음"
        return (
            f"{nt['company']} {nt['account_name']} 전년 동기 {prior:,}원 → 당기 {current:,}원 "
            f"({truth:+.2f}%, 실제 방향: {direction})"
        )
    if kind == "per":
        truth = _per_ground_truth(nt)
        if truth is None:
            return None
        return f"{nt['company']} {nt['on']} 기준 {nt.get('field', 'PER')} {truth:.2f}배"
    if kind == "peer_top":
        truth = _peer_top_ground_truth(nt)
        if truth is None:
            return None
        return f"{nt['group']} 업종 {nt['metric']} 1위: {truth[0]} ({truth[1]:.2f}%, 각 회사 가장 최근 보고서 누적 기준)"
    if kind == "divergence":
        current, prior = _growth_amounts(nt["growth"])
        growth_truth = _growth_ground_truth(nt["growth"])
        price_truth = _price_return_ground_truth(nt["price"])
        if growth_truth is None or price_truth is None:
            return None
        growth_dir = "증가" if growth_truth > 0 else "감소" if growth_truth < 0 else "변화 없음"
        price_dir = "상승" if price_truth > 0 else "하락" if price_truth < 0 else "보합"
        g = nt["growth"]
        return (
            f"{g['company']} {g['account_name']} 전년 동기 {prior:,}원 → 당기 {current:,}원 ({growth_truth:+.2f}%, {growth_dir}), "
            f"같은 기간({nt['price']['start']}~{nt['price']['end']}) 주가 {price_truth:+.2f}% ({price_dir})"
        )
    raise ValueError(f"알 수 없는 numeric_target kind: {kind}")


# 답변 모델(nano)보다 한 단계 위 모델. nano 심판은 사람 판정과 일치율 62~81%에 그쳤고, 조회 자료가
# 붙어 입력이 길어지면 채점 기준을 스스로 인용하고도 어기는 판정이 나왔다 (2026-09-24 메타평가).
JUDGE_MODEL = "gpt-5.4-mini"

_judge_client = None
# 심판 응답 캐시 (API 절약): 모델+프롬프트 전문이 같으면 저장된 응답을 재사용한다. 프롬프트에 답변·채점 기준·
# 참고 정답·조회 자료가 전부 들어가므로, 이 중 하나라도 바뀌면 키가 달라져 자동으로 새로 판정한다.
# JSONL(한 줄에 한 항목)로 추가만 한다 - 예전엔 dict 전체를 매번 다시 썼는데(항목이 늘수록 저장 1회
# 비용이 계속 커지는 O(n²)), 이제는 새 항목 한 줄만 추가한다 (2026-09-29).
_JUDGE_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "judge_cache_export.jsonl"
_LEGACY_JUDGE_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "judge_cache_export.json"
_judge_cache: dict[str, str] | None = None


def _get_judge_client() -> OpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAI()
    return _judge_client


def _load_judge_cache() -> dict[str, str]:
    cache: dict[str, str] = {}
    if _LEGACY_JUDGE_CACHE_PATH.exists():  # 예전 형식(전체를 다시 쓰던 dict) - 있으면 1회만 읽어 이어받는다
        cache.update(json.loads(_LEGACY_JUDGE_CACHE_PATH.read_text(encoding="utf-8")))
    if _JUDGE_CACHE_PATH.exists():
        with open(_JUDGE_CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cache[entry["key"]] = entry["value"]
    return cache


def _judge_json(prompt: str) -> str:
    """심판 LLM을 JSON 모드로 호출하고 원문 응답을 반환 (캐시 적중 시 API 호출 없음)."""
    global _judge_cache
    if _judge_cache is None:
        _judge_cache = _load_judge_cache()
    key = hashlib.sha256(f"{JUDGE_MODEL}\n{prompt}".encode("utf-8")).hexdigest()
    if key not in _judge_cache:
        resp = _get_judge_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        value = resp.choices[0].message.content
        _judge_cache[key] = value
        _JUDGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_JUDGE_CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
    return _judge_cache[key]


def describe_numeric_check(numeric_grade: dict) -> str | None:
    """규칙 기반 수치 채점 결과를 심판에게 넘길 문장으로 바꾼다 (채점 불가면 None)."""
    if not numeric_grade["applicable"] or numeric_grade["correct"] is None:
        return None
    if numeric_grade["correct"]:
        return "답변에 참고 정답과 일치하는 수치가 있음 (조/억 단위 환산까지 코드로 확인 완료)"
    return "답변에 참고 정답과 일치하는 수치가 없음 (조/억 단위 환산까지 코드로 확인 완료)"


def judge_answer(
    question: str,
    rubric: str,
    answer: str,
    reference: str | None,
    numeric_check: str | None = None,
    evidence: str | None = None,
) -> dict:
    """LLM 심판: 답변이 채점 기준을 충족하는지 pass/fail + 근거로 판정한다.

    규칙 기반 채점(숫자가 텍스트에 있는지)은 "결론이 서로 모순되는지", "질문의 핵심에
    실제로 답했는지"를 못 본다(q10 자기모순, q5 되묻기 사례). 그걸 채점 기준표로 판정.
    반대로 nano 심판은 "305조 3,729억"을 원 단위와 대조하는 계산을 틀리므로(q1 오판 사례),
    수치 일치 여부는 코드 결과(numeric_check)를 넘겨주고 심판은 수치 외 기준만 본다.
    evidence(답변자가 도구로 조회한 자료)가 없으면 심판은 "그 원인이 실제 기사에 있었는지"를
    검증할 수 없어 근거 있는 답변도 떨어뜨렸다(q4 오판 사례) — 그래서 함께 넘겨서 근거
    충실성(환각 여부)까지 판단하게 한다.
    """
    numeric_line = (
        f"[수치 검사 결과 (코드로 이미 확인함 - 그대로 믿을 것)]\n{numeric_check}\n\n" if numeric_check else ""
    )
    evidence_line = f"[답변자가 도구로 조회한 자료]\n{evidence}\n\n" if evidence else ""
    prompt = (
        "너는 금융 리서치 답변을 채점하는 엄격한 채점자야. 아래 [답변]이 [채점 기준]을 모두 "
        "충족하는지 판정해.\n\n"
        f"[질문]\n{question}\n\n"
        f"[채점 기준]\n{rubric}\n\n"
        f"[참고 정답 (DB/거래소에서 조회한 사실. 답변자에게는 주어지지 않았음)]\n{reference or '없음 - 채점 기준에 따라 판단'}\n\n"
        f"{numeric_line}"
        f"{evidence_line}"
        f"[답변]\n{answer}\n\n"
        "판정 규칙:\n"
        "- 채점 기준을 모두 충족할 때만 pass=true. 채점 기준에 없는 요구사항을 추가로 만들지 마.\n"
        "- [수치 검사 결과]가 주어지면 수치 일치 여부는 직접 계산하지 말고 그 결과를 따라. "
        "너는 수치 외의 기준(결론의 정확성·일관성, 원인 설명, 전제 교정 등)만 판단해. "
        "수치 검사 결과가 '일치'인데 단위 중복('만원 원') 같은 표기 오타만 있다면 감점 사유가 아니다.\n"
        "- 답변이 원인·사실의 근거로 든 내용은 [답변자가 도구로 조회한 자료]에 실제로 있는지 대조해. "
        "자료에 있으면 근거로 인정하고, 자료에 없는 내용을 사실처럼 단정했으면(환각) fail. "
        "자료의 기사 내용은 앞뒤가 잘린 요약 조각이라 문장이 중간에서 시작하거나 끝날 수 있으니, "
        "답변의 수치·표현이 그 조각 안에 나오면 근거로 인정해.\n"
        "- 답변 안에서 결론끼리 서로 모순되면 fail.\n"
        "- 핵심 요구를 충족하지 못한 채 사용자에게 되묻기만 하거나 '모른다'로 끝나면 fail. "
        "핵심 요구를 이미 충족했다면 끝에 추가 질문을 덧붙인 것은 fail 사유가 아니다.\n"
        'JSON으로만 답해: {"pass": true 또는 false, "reason": "판정 근거 한두 문장"}'
    )
    raw = _judge_json(prompt)
    try:
        verdict = json.loads(raw)
        passed = verdict.get("pass")
        return {"pass": passed if isinstance(passed, bool) else None, "reason": verdict.get("reason", ""), "raw": raw}
    except json.JSONDecodeError:
        return {"pass": None, "reason": f"심판 응답 JSON 파싱 실패: {raw[:200]}", "raw": raw}


def _pairwise_once(
    question: str, rubric: str, reference: str | None, first: dict, second: dict
) -> dict:
    prompt = (
        "너는 금융 리서치 답변 두 개를 비교하는 엄격한 채점자야. 질문에 더 정확하고 유용하게 답한 쪽을 골라.\n\n"
        f"[질문]\n{question}\n\n"
        f"[채점 기준]\n{rubric}\n\n"
        f"[참고 정답 (DB/거래소에서 조회한 사실)]\n{reference or '없음 - 채점 기준에 따라 판단'}\n\n"
        f"[답변 A]\n{first['answer']}\n\n[답변 A 수치 검사 결과 (코드로 확인)]\n{first['numeric_check'] or '해당 없음'}\n\n"
        f"[답변 B]\n{second['answer']}\n\n[답변 B 수치 검사 결과 (코드로 확인)]\n{second['numeric_check'] or '해당 없음'}\n\n"
        "판정 규칙:\n"
        "- 채점 기준 충족 여부를 가장 먼저 보고, 둘 다 충족하거나 둘 다 못 하면 정확성·근거의 구체성·일관성으로 비교해.\n"
        "- 수치 일치 여부는 직접 계산하지 말고 수치 검사 결과를 따라.\n"
        "- 길이나 말투가 아니라 내용으로 판단해. 차이가 거의 없으면 tie.\n"
        'JSON으로만 답해: {"winner": "A" 또는 "B" 또는 "tie", "reason": "판정 근거 한두 문장"}'
    )
    raw = _judge_json(prompt)
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        return {"winner": None, "reason": f"심판 응답 JSON 파싱 실패: {raw[:200]}"}
    winner = verdict.get("winner")
    return {"winner": winner if winner in ("A", "B", "tie") else None, "reason": verdict.get("reason", "")}


def pairwise_judge(question: str, rubric: str, reference: str | None, x: dict, y: dict) -> dict:
    """두 답변(x, y) 중 나은 쪽을 고른다. x/y = {"name", "answer", "numeric_check"}.

    LLM 심판은 먼저 나온 답을 선호하는 위치 편향이 있어서, 순서를 바꿔 두 번 묻고 두 판정이
    같은 답변을 가리킬 때만 승자로 인정한다. 엇갈리면 tie.
    """
    xy = _pairwise_once(question, rubric, reference, x, y)  # A=x, B=y
    yx = _pairwise_once(question, rubric, reference, y, x)  # A=y, B=x
    pick_xy = {"A": x["name"], "B": y["name"], "tie": "tie"}.get(xy["winner"])
    pick_yx = {"A": y["name"], "B": x["name"], "tie": "tie"}.get(yx["winner"])
    winner = pick_xy if pick_xy == pick_yx and pick_xy is not None else "tie"
    return {
        "winner": winner,
        "consistent": pick_xy == pick_yx,
        "order1": {"pick": pick_xy, "reason": xy["reason"]},
        "order2": {"pick": pick_yx, "reason": yx["reason"]},
    }


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
