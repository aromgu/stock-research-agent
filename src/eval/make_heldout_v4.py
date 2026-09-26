"""held-out v4 문항 생성기: 사람이 문항을 고르지 않고, 규칙과 고정 시드 난수로 만든다.

v1~v3는 에이전트를 고치는 쪽이 문항을 직접 써서, 무의식중에 에이전트가 잘하거나 못하는 조합이 들어갈 수 있었다.
v4의 오염 방지 장치:
1. 질문 유형(9종)과 문장 틀은 실제 사용자가 물을 법한 형태로 먼저 정하고, 회사·기간·지표·문장 틀·전제의 참/거짓은
   고정 시드 난수가 고른다. 특정 조합을 사람이 골라 넣거나 뺄 수 없다.
2. 개발 내내 보던 삼성전자·SK하이닉스는 대상 회사에서 뺀다 (튜닝 셋과 프롬프트 예시가 이 두 회사 위주).
3. 전제의 참/거짓과 업종 비교 그룹 종류는 유형 안에서 번갈아 배정해 한쪽으로 쏠리지 않게 한다.
4. 버리는 조합은 정답을 계산할 수 없는 경우뿐이다 (데이터 없음, 전년 적자로 증가율 정의 불가, 휴장일,
   방향이 모호한 ±2% 이내 주가 변동 등). 규칙은 코드에 있다.
5. 만든 파일의 SHA-256을 gold_set.py에 고정한다. 결과를 본 뒤 문항을 고치면 로드할 때 바로 드러난다.

실행: python -m src.eval.make_heldout_v4  (같은 시드면 같은 문항이 나온다 — 데이터가 같다는 전제)
"""

import calendar
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

from ..agent import tools
from ..data import dart_client as dc
from ..data import universe as uv
from . import graders as g

SEED = 20260926
PER_TYPE = 5
OUT_PATH = Path(__file__).with_name("heldout_v4.json")
_EXCLUDED = {"삼성전자", "SK하이닉스"}
_INFORMAL = {official: informal for informal, official in dc._ALIASES.items()}  # 현대자동차 → 현대차 등

# 손익 항목 기간: (질문 문구, 사업연도, 보고서 코드, 당기 금액 필드, 전년 동기 금액 필드, 비교 문구)
_FLOW_PERIODS = [
    ("2026년 상반기(1~6월) 누적", "2026", "11012", "thstrm_add_amount", "frmtrm_add_amount", "전년 동기"),
    ("2026년 1분기", "2026", "11013", "thstrm_amount", "frmtrm_amount", "전년 동기"),
    ("2025년 연간", "2025", "11011", "thstrm_amount", "frmtrm_amount", "전년"),
    ("2025년 1~9월 누적", "2025", "11014", "thstrm_add_amount", "frmtrm_add_amount", "전년 동기"),
]
# 재무상태 항목 시점: (질문 문구, 사업연도, 보고서 코드)
_STOCK_PERIODS = [
    ("2026년 6월 말", "2026", "11012"),
    ("2026년 3월 말", "2026", "11013"),
    ("2025년 말", "2025", "11011"),
    ("2025년 9월 말", "2025", "11014"),
]
_NET_INCOME = "당기순이익(손실)"
_SAY = {_NET_INCOME: "순이익"}
_MONTHS = list(range(1, 9))  # 2026년 1~8월 (완전히 끝난 달만)


def _say_company(name: str, rng: random.Random) -> str:
    """사람들이 흔히 쓰는 줄임말이 있으면 절반은 그걸로 묻는다 (현대차, 네이버 등)."""
    return _INFORMAL[name] if name in _INFORMAL and rng.random() < 0.5 else name


def _month_window(m: int) -> tuple[str, str]:
    return f"2026-{m:02d}-01", f"2026-{m:02d}-{calendar.monthrange(2026, m)[1]:02d}"


def _q(qtype: str, question: str, rubric: str, company: str, sources: list[str], nt: dict) -> dict:
    return {"type": qtype, "question": question, "rubric": rubric, "company": company,
            "expected_sources": sources, "numeric_target": nt}  # fmt: skip


def gen_amount(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    c = rng.choice(companies)
    label, y, r, cur, _, _ = rng.choice(_FLOW_PERIODS)
    acct = rng.choice(["매출액", "영업이익", _NET_INCOME])
    nt = {"company": c, "bsns_year": y, "reprt_code": r, "account_name": acct, "amount_field": cur}
    if g.get_ground_truth(nt) is None:
        return None
    a = _SAY.get(acct, acct)
    who = _say_company(c, rng)
    question = rng.choice([f"{who} {label} {a}은 얼마야?", f"{who}의 {label} {a} 알려줘", f"{label} 기준으로 {who} {a}이 얼마였어?"])
    rubric = f"{c}의 {label} {a}을 구체적 금액으로 제시해야 한다. 다른 기간(분기 단독 등)의 값을 결론으로 제시하거나 수치를 제시하지 못하면 fail."
    return _q("amount", question, rubric, c, ["dart"], nt)


def gen_ratio(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    c = rng.choice(companies)
    metric = rng.choice(["영업이익률", "순이익률", "부채비율"])
    if metric == "부채비율":
        label, y, r = rng.choice(_STOCK_PERIODS)
        num, den, field = "부채총계", "자본총계", "thstrm_amount"
    else:
        label, y, r, field, _, _ = rng.choice(_FLOW_PERIODS)
        num, den = ("영업이익" if metric == "영업이익률" else _NET_INCOME), "매출액"
    nt = {"kind": "ratio", "company": c, "bsns_year": y, "reprt_code": r,
          "numerator_account": num, "denominator_account": den, "amount_field": field}  # fmt: skip
    truth = g._ratio_ground_truth(nt)
    if truth is None or (metric == "부채비율" and truth <= 0):
        return None
    who = _say_company(c, rng)
    question = rng.choice([f"{who}의 {label} 기준 {metric}은 몇 %야?", f"{who} {label} {metric} 알려줘"])
    rubric = f"{c}의 {label} 기준 {metric}({_SAY.get(num, num)} ÷ {den})을 %로 제시해야 한다. 다른 기간 값을 제시하거나 수치가 없으면 fail."
    return _q("ratio", question, rubric, c, ["dart"], nt)


def _growth_target(rng: random.Random, companies: list[str], min_abs_change: float):
    c = rng.choice(companies)
    label, y, r, cur, prior, versus = rng.choice(_FLOW_PERIODS)
    acct = rng.choice(["매출액", "영업이익"])
    nt = {"kind": "growth", "company": c, "bsns_year": y, "reprt_code": r, "account_name": acct,
          "current_field": cur, "prior_field": prior}  # fmt: skip
    current, prior_amount = g._growth_amounts(nt)
    if current is None or not prior_amount or prior_amount <= 0:  # 전년 적자면 증가율이 정의되지 않음
        return None
    change = current / prior_amount - 1
    if abs(change) < min_abs_change:
        return None
    return c, label, acct, versus, nt, change


def gen_growth(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    picked = _growth_target(rng, companies, min_abs_change=0.0)
    if picked is None:
        return None
    c, label, acct, versus, nt, _ = picked
    who = _say_company(c, rng)
    question = rng.choice(
        [f"{who} {label} {acct}은 {versus} 대비 얼마나 변했어?", f"{who}의 {label} {acct} 증감률이 {versus} 대비 몇 %야?"]
    )
    rubric = f"{c}의 {label} {acct}이 {versus} 대비 몇 % 늘었는지/줄었는지를 증감률 또는 두 기간 금액으로 제시하고 방향을 올바르게 말해야 한다. 방향이 틀리거나 다른 기간끼리 비교하면 fail."
    return _q("growth", question, rubric, c, ["dart"], nt)


def gen_compare(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    groups = [
        [n for n in members if n not in _EXCLUDED] for members in uv.SEMICONDUCTOR_SEGMENTS.values()
    ] + [[c["name"] for c in uv.load_universe() if uv.KOSPI_TOP_LABEL in c["groups"] and c["name"] not in _EXCLUDED]]
    group = rng.choice([grp for grp in groups if len(grp) >= 2])
    a, b = rng.sample(sorted(group), 2)
    metric = rng.choice(["영업이익률", "부채비율"])
    if metric == "부채비율":
        label, y, r = rng.choice(_STOCK_PERIODS)
        num, den, field = "부채총계", "자본총계", "thstrm_amount"
    else:
        label, y, r, field, _, _ = rng.choice(_FLOW_PERIODS)
        num, den = "영업이익", "매출액"

    def target(company: str) -> dict:
        return {"company": company, "bsns_year": y, "reprt_code": r, "numerator_account": num,
                "denominator_account": den, "amount_field": field}  # fmt: skip

    ta, tb = g._ratio_ground_truth(target(a)), g._ratio_ground_truth(target(b))
    if ta is None or tb is None or abs(ta - tb) < 1.0:  # 1%p 미만 차이는 "어느 쪽이 높나"가 모호
        return None
    higher = rng.random() < 0.5
    winner = (a if ta > tb else b) if higher else (a if ta < tb else b)
    word = "높은" if higher else "낮은"
    question = f"{_say_company(a, rng)}, {_say_company(b, rng)} 두 회사 중 {label} 기준 {metric}이 더 {word} 곳은 어디야? 수치도 알려줘"
    rubric = f"두 회사의 {label} 기준 {metric}을 각각 %로 제시하고, 더 {word} 곳({winner})을 올바르게 결론 내려야 한다. 결론이 틀리거나 수치가 없으면 fail."
    nt = {"kind": "ratio_pair", "company_a": target(a), "company_b": target(b)}
    return {**_q("compare", question, rubric, a, ["dart"], nt), "dedupe_key": ("compare", frozenset((a, b)), metric)}


def gen_price_return(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    c = rng.choice(companies)
    m = rng.choice(_MONTHS)
    start, end = _month_window(m)
    nt = {"kind": "price_return", "company": c, "start": start, "end": end}
    if g._price_return_ground_truth(nt) is None:
        return None
    who = _say_company(c, rng)
    question = rng.choice([f"{who} 주가는 2026년 {m}월 한 달 동안 몇 % 움직였어?", f"2026년 {m}월 {who} 주가 수익률이 어떻게 돼?"])
    rubric = f"{c}의 2026년 {m}월 주가 수익률(그 달 첫 거래일 종가 대비 마지막 거래일 종가)을 %로 제시하고 방향을 올바르게 말해야 한다. 다른 기간 값을 제시하거나 수치가 없으면 fail."
    return _q("price_return", question, rubric, c, ["price"], nt)


def gen_valuation(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    c = rng.choice(companies)
    field = rng.choice(["PER", "PBR"])
    day = date(2026, 9, 1) + timedelta(days=rng.randrange(23))
    if day.weekday() >= 5:
        return None
    nt = {"kind": "per", "field": field, "company": c, "on": day.isoformat()}
    truth = g._per_ground_truth(nt)
    if truth is None or truth <= 0:  # 적자 회사는 PER이 정의되지 않음 / 휴장일
        return None
    say_day = f"2026년 9월 {day.day}일"
    who = _say_company(c, rng)
    question = rng.choice([f"{who}의 {say_day} 기준 {field}는 몇 배야?", f"{say_day} 기준 {who} {field} 알려줘"])
    rubric = f"{c}의 {say_day} 기준 {field}를 배수로 제시해야 한다. 다른 날짜 값을 제시하거나 수치가 없으면 fail."
    return _q("valuation", question, rubric, c, ["price"], nt)


def gen_price_premise(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    c = rng.choice(companies)
    m = rng.choice(_MONTHS)
    start, end = _month_window(m)
    nt = {"kind": "price_direction", "company": c, "start": start, "end": end}
    ret = g._price_return_ground_truth(nt)
    if ret is None or abs(ret) < 2.0:  # 거의 안 움직인 달은 "올랐다/내렸다"가 모호
        return None
    actual_up = ret > 0
    premise_true = slot % 2 == 0  # 맞는 전제와 틀린 전제를 번갈아 (난수로 두면 5문항이 한쪽으로 쏠릴 수 있음)
    asks_up = actual_up if premise_true else not actual_up
    verb = "올랐어" if asks_up else "떨어졌어"
    question = f"{_say_company(c, rng)} 주가가 2026년 {m}월에 왜 {verb}?"
    actual = f"실제로는 {'상승' if actual_up else '하락'}({ret:+.1f}%)했다"
    rubric = (
        f"{actual}. 전제가 맞으면 주가 데이터로 사실임을 확인하고, 틀리면 답변 첫머리에서 바로잡은 뒤 실제 방향을 기준으로 "
        "설명해야 한다. 원인은 조회 자료의 구체적 사실로 뒷받침해야 하며, 못 찾은 부분은 솔직히 밝히면 된다. "
        "틀린 전제를 받아들이거나, 맞는 전제를 틀렸다고 하거나, 근거 없는 원인을 사실처럼 단정하면 fail."
    )
    return _q("price_premise", question, rubric, c, ["price", "news"], nt)


def gen_financial_premise(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    picked = _growth_target(rng, companies, min_abs_change=0.03)
    if picked is None:
        return None
    c, label, acct, versus, nt, change = picked
    premise_true = slot % 2 == 0
    asks_up = (change > 0) if premise_true else not (change > 0)
    verb = "늘었어" if asks_up else "줄었어"
    question = f"{_say_company(c, rng)} {label} {acct}이 왜 {verb}?"
    actual = f"실제로는 {versus} 대비 {'증가' if change > 0 else '감소'}({change * 100:+.1f}%)했다"
    rubric = (
        f"{actual}. 두 기간의 {acct} 금액이나 증감률을 제시하고, 전제가 맞으면 확인, 틀리면 답변 첫머리에서 바로잡아야 한다. "
        "원인은 조회 자료의 구체적 사실로 뒷받침해야 하며, 못 찾은 부분은 솔직히 밝히면 된다. "
        "틀린 전제를 받아들이거나, 맞는 전제를 틀렸다고 하거나, 근거 없는 원인을 사실처럼 단정하면 fail."
    )
    return _q("financial_premise", question, rubric, c, ["dart", "news"], nt)


_PEER_METRICS = ["영업이익률", "순이익률", "매출액 증가율", "영업이익 증가율"]


def gen_peer_top(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    choice = ("segment", "kospi", "company")[slot % 3]  # 비교 그룹 종류를 돌아가며
    metric = rng.choice(_PEER_METRICS)
    if choice == "segment":
        seg = rng.choice([s for s, members in uv.SEMICONDUCTOR_SEGMENTS.items() if len(members) >= 3])
        group, subject = seg, f"반도체 {seg} 업체들"
    elif choice == "kospi":
        group, subject = uv.KOSPI_TOP_LABEL, "코스피 시가총액 상위 30개 회사"
    else:
        c = rng.choice(companies)
        group, subject = c, f"{_say_company(c, rng)}의 동종 업계 회사들"
    _, members, _ = tools._resolve_peer_group(group)
    if len(members) < 3:
        return None
    nt = {"kind": "peer_top", "group": group, "metric": metric}
    truth = g._peer_top_ground_truth(nt)
    if truth is None:
        return None
    question = f"{subject} 중 2026년 상반기 누적 기준 {metric}이 가장 높은 곳은 어디고 몇 %야?"
    rubric = f"비교 대상 업종에서 {metric} 1위 회사와 그 수치를 제시해야 한다. 1위 회사를 틀리게 말하거나 수치가 없으면 fail."
    representative = next(m["name"] for m in members if m["name"] not in _EXCLUDED)  # 고정 파이프라인이 쓸 대표 회사
    return _q("peer_top", question, rubric, representative, ["dart"], nt)


GENERATORS = [
    ("amount", gen_amount),
    ("ratio", gen_ratio),
    ("growth", gen_growth),
    ("compare", gen_compare),
    ("price_return", gen_price_return),
    ("valuation", gen_valuation),
    ("price_premise", gen_price_premise),
    ("financial_premise", gen_financial_premise),
    ("peer_top", gen_peer_top),
]


def generate(seed: int = SEED, per_type: int = PER_TYPE) -> list[dict]:
    rng = random.Random(seed)
    companies = sorted(c["name"] for c in uv.load_universe() if c["name"] not in _EXCLUDED)
    questions, seen = [], set()
    for qtype, gen in GENERATORS:
        made, attempts = 0, 0
        while made < per_type:
            attempts += 1
            if attempts > 400:
                raise RuntimeError(f"{qtype}: 정답을 계산할 수 있는 조합을 {per_type}개 못 찾음")
            q = gen(rng, companies, made)
            if q is None:
                continue
            key = q.pop("dedupe_key", q["question"])  # 비교 문항은 같은 회사 쌍·지표면 기간만 달라도 중복으로 봄
            if key in seen:
                continue
            seen.add(key)
            made += 1
            questions.append({"id": f"v4_{qtype}_{made}", **q})
    return questions


def main() -> None:
    questions = generate()
    payload = {"seed": SEED, "generator": "src/eval/make_heldout_v4.py", "questions": questions}
    text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for q in questions:
        print(f"{q['id']:<22} {q['question']}")
    print(f"\n{len(questions)}문항 → {OUT_PATH}\nSHA-256: {digest}")


if __name__ == "__main__":
    main()
