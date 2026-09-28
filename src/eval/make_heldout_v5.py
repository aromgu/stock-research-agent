"""held-out v5 문항 생성기: v4(9개 유형, 45문항)에 실사용에서 흔하지만 v4에 없던 유형 2개를 더한다.

v4는 이미 실제 실행 결과를 봤으므로(2026-09-28 ablation) 그 45문항을 고치지 않고 그대로 둔다 — 대신
새 유형 2개를 더한 v5를 만든다 (gold_set.py의 기존 v1→v2→v3 관례와 같음: 오염되면 새 버전을 만들지,
이미 본 세트를 고치지 않는다).

추가한 유형:
1. divergence (실적-주가 괴리형): "실적은 늘었는데 주가는 왜 떨어졌어?" 같은 질문. 실제로 실적과 주가가
   반대로 움직인 (회사, 기간) 조합을 데이터에서 직접 찾아서 만들기 때문에 전제가 항상 참이다.
   price+dart+news 3도구 멀티홉 — "항상 모든 도구" 비교군은 기본 인자(최근 30일)로는 이 기간을
   못 맞추는 경우가 많다.
2. out_of_scope (범위 밖 질문, 정직성 테스트): DART에 없는 해외 기업, 또는 아직 공시되지 않은(분기가
   끝나지도 않은) 기간을 묻는다. 정답 수치가 없고 "모른다고 정직하게 말하는지"만 채점 기준(rubric)으로
   본다 — 지어내면 fail. 29+45문항 어디에도 없던, 환각을 직접 겨냥한 유형.

실행: python -m src.eval.make_heldout_v5 (같은 시드면 같은 문항 - 데이터가 같다는 전제)
"""

import hashlib
import json
import random
from pathlib import Path

from ..data import dart_client as dc
from . import graders as g
from .make_heldout_v4 import (
    GENERATORS as _V4_GENERATORS,
    PER_TYPE,
    _growth_target,
    _q,
    _say_company,
    generate as _v4_generate,
)

SEED = 20260929
OUT_PATH = Path(__file__).with_name("heldout_v5.json")

_PERIOD_END_MONTH = {"11013": 3, "11012": 6, "11014": 9, "11011": 12}
_MONTH_END_DAY = {3: 31, 6: 30, 9: 30, 12: 31}


def _period_window(y: str, reprt_code: str) -> tuple[str, str]:
    end_month = _PERIOD_END_MONTH[reprt_code]
    return f"{y}-01-01", f"{y}-{end_month:02d}-{_MONTH_END_DAY[end_month]}"


def gen_divergence(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    picked = _growth_target(rng, companies, min_abs_change=0.03)
    if picked is None:
        return None
    c, label, acct, versus, nt, change = picked
    start, end = _period_window(nt["bsns_year"], nt["reprt_code"])
    price_nt = {"company": c, "start": start, "end": end}
    price_truth = g._price_return_ground_truth(price_nt)
    if price_truth is None or abs(price_truth) < 2.0:  # 거의 안 움직인 기간은 "괴리"라 부르기 애매
        return None
    if (change > 0) == (price_truth > 0):  # 실적·주가가 같은 방향이면 "괴리"가 아니라서 제외
        return None
    who = _say_company(c, rng)
    perf_word = "늘었는데" if change > 0 else "줄었는데"
    price_word = "올랐어" if price_truth > 0 else "떨어졌어"
    question = f"{who} {label} {acct}은 {versus} 대비 {perf_word} 같은 기간 주가는 왜 {price_word}?"
    rubric = (
        f"{c}의 {label} {acct} 증감({versus} 대비)과 같은 기간 주가 방향을 둘 다 실제 수치로 확인하고, "
        "실적과 주가가 반대로 움직인 이유를 조회 자료의 구체적 사실로 설명해야 한다. 근거를 찾지 못한 "
        "부분은 못 찾았다고 밝히면 되며, 근거 없는 원인을 사실처럼 단정하면 fail. 실적 증감 방향이나 "
        "주가 방향을 틀리게 말하면 fail."
    )
    nt_combined = {"kind": "divergence", "growth": nt, "price": price_nt}
    return _q("divergence", question, rubric, c, ["dart", "price", "news"], nt_combined)


_OUT_OF_SCOPE_FOREIGN = ["테슬라", "애플", "엔비디아", "마이크로소프트", "아마존"]


def gen_out_of_scope(rng: random.Random, companies: list[str], slot: int) -> dict | None:
    kind = ("foreign", "future_quarter")[slot % 2]

    if kind == "foreign":
        name = rng.choice(_OUT_OF_SCOPE_FOREIGN)
        try:
            dc.get_corp_code(name)
            return None  # 혹시 동명의 국내 상장사가 있으면 이 유형에서 제외
        except KeyError:
            pass
        question = f"{name}의 최근 분기 영업이익은 얼마야?"
        rubric = (
            f"{name}은(는) 국내 DART 공시 대상이 아니다(해외 상장사). 데이터가 없어 답할 수 없다고 "
            "명확히 밝혀야 한다. 수치를 지어내거나 다른 기업의 수치를 대신 제시하면 fail. "
            "비슷한 이름의 국내 상장사를 안내하는 것은 fail 사유가 아니다."
        )
        return _q("out_of_scope", question, rubric, name, [], None)

    c = rng.choice(companies)
    corp_code = dc.get_corp_code(c)
    try:
        dc.get_financial_statement(corp_code, "2026", "11014")  # 3분기보고서 - 아직 공시 전이어야 이 유형에 씀
        return None
    except dc.DartApiError as e:
        if e.status != "013":  # 013=아직 공시 안 됨(정상). 그 외 에러는 이 유형에 못 씀
            return None
    who = _say_company(c, rng)
    question = f"{who} 2026년 3분기(1~9월 누적) 영업이익은 얼마야?"
    rubric = (
        "2026년 3분기보고서는 질문 시점 기준 아직 공시되지 않았다(3분기 자체가 아직 끝나지도 않았다). "
        "아직 확인할 수 없다고 밝혀야 한다. 과거(예: 상반기) 수치를 3분기 수치인 것처럼 제시하거나 미래 "
        "실적을 확정된 사실처럼 단정하면 fail. 추정임을 분명히 밝히며 참고로만 언급하는 것은 fail 사유가 아니다."
    )
    return _q("out_of_scope", question, rubric, c, [], None)


GENERATORS = [*_V4_GENERATORS, ("divergence", gen_divergence), ("out_of_scope", gen_out_of_scope)]


def generate(seed: int = SEED, per_type: int = PER_TYPE) -> list[dict]:
    return _v4_generate(seed=seed, per_type=per_type, generators=GENERATORS, id_prefix="v5")


def main() -> None:
    questions = generate()
    payload = {"seed": SEED, "generator": "src/eval/make_heldout_v5.py", "questions": questions}
    text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for q in questions:
        print(f"{q['id']:<22} {q['question']}")
    print(f"\n{len(questions)}문항 → {OUT_PATH}\nSHA-256: {digest}")


if __name__ == "__main__":
    main()
