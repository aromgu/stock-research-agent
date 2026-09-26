"""Phase 5: 채점용 골드 질문셋.

각 질문에 "이 출처가 인용돼야 한다"는 정답 라벨(expected_sources)과,
숫자로 채점 가능한 질문이면 DART DB에서 정답값을 찾기 위한 좌표(numeric_target)를 붙인다.
숫자 정답 자체는 하드코딩하지 않고 채점 시점에 DB에서 직접 조회한다 (데이터가 바뀌어도 안 썩게).

split:
- "tuning" (TUNING_QUESTIONS): 프롬프트·채점기를 고칠 때 실제로 보고 고친 질문. 여기 점수는 과대평가될 수 있다.
- "heldout" (HELDOUT_QUESTIONS): 2026-09-24 결과를 보기 전에 확정한 질문. 이 질문들의 결과를 보고
  프롬프트나 채점 기준을 고치면 held-out의 의미가 사라지므로 고치지 말 것 (고쳤다면 새 held-out을 만들 것).
  → 실제로 h5를 보고 재무 도구 강제와 증가율 채점을 고쳐서 일부 오염됨.
- "heldout_v2" (HELDOUT_V2_QUESTIONS): v1이 오염된 뒤 새로 만든 셋. 이후 연도 지정 결함을 v2에서 발견해 고침.
- "heldout_v3" (HELDOUT_V3_QUESTIONS): 지원 종목 확장·업종 비교·답변 모델 분리 이후 만든 셋. 새 기능을 직접 묻는다.
- "heldout_v4" (HELDOUT_V4_QUESTIONS): 사람이 고르지 않고 make_heldout_v4.py가 규칙과 고정 시드로 만든 45문항.
  파일 해시를 아래에 고정해 결과를 본 뒤 고치면 로드가 실패한다.
"""

import hashlib
import json
from pathlib import Path

TUNING_QUESTIONS = [
    {
        "id": "q1_dart_revenue",
        # "반기보고서 기준 매출액"은 모호함(해당 분기 단독 171조 vs 상반기 누적 305조 둘 다
        # 말이 됨) — LLM이 어느 쪽으로 답해도 자연스러워서 채점이 운에 좌우됐다.
        # "1~6월 누적"으로 못박아 thstrm_add_amount와 짝을 맞춘다.
        "question": "삼성전자 2026년 1~6월(상반기) 누적 매출액이 얼마야?",
        "rubric": "2026년 상반기(1~6월) 누적 매출액을 구체적 금액으로 제시해야 한다. 분기 단독 금액 등 다른 값을 결론으로 제시하거나 수치를 제시하지 못하면 fail.",
        "company": "삼성전자",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "삼성전자",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "매출액",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "q2_dart_operating_profit",
        "question": "SK하이닉스 2026년 1~6월(상반기) 누적 영업이익이 얼마야?",
        "rubric": "2026년 상반기(1~6월) 누적 영업이익을 구체적 금액으로 제시해야 한다. 분기 단독 금액 등 다른 값을 결론으로 제시하거나 수치를 제시하지 못하면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "SK하이닉스",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "영업이익",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "q3_dart_total_assets",
        "question": "SK하이닉스 최근 반기보고서 기준 자산총계는 얼마야?",
        "rubric": "2026년 반기말(6월 말) 기준 자산총계를 구체적 금액으로 제시해야 한다. 수치를 제시하지 못하면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "SK하이닉스",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "자산총계",
        },
    },
    {
        "id": "q4_news_stock_price",
        # Price 도구가 생기기 전에는 news만으로 때웠지만, 이제 "정말 올랐는지"부터
        # price로 확인하고 나서 news로 원인을 설명하는 게 제대로 된 답변이다.
        "question": "삼성전자 최근에 주가가 왜 올랐어?",
        "rubric": "최근 주가가 실제로 올랐는지 근거(주가 데이터 또는 기사)로 확인하고, 상승 원인을 구체적 근거와 함께 최소 1가지 설명해야 한다. 기사에서 나온 구체적 사실(날짜·이벤트·수치 등)이 제시되면 근거로 인정하며, 기사 제목이나 매체명까지는 요구하지 않는다. 원인 없이 되묻거나 모른다고만 하면 fail.",
        "company": "삼성전자",
        "expected_sources": {"price", "news"},
        "numeric_target": None,  # "최근"이 모호해서 자동 채점 가능한 고정 구간이 아님
    },
    {
        "id": "q5_multihop_samsung",
        "question": "삼성전자 2026년 상반기 누적 실적이 왜 이렇게 나왔는지 영업이익 숫자랑 같이 설명해줘",
        # 예전 기준("서로 다른 원인 최소 2가지")은 근거가 원인 하나만 뒷받침할 때 두 번째 원인을
        # 지어내야 통과하는 구조라 환각을 부추겼다 → 개수 대신 근거와 정직성을 본다 (2026-09-24 수정)
        "rubric": "① 2026년 상반기 누적 영업이익 수치 ② 그 실적이 나온 원인을 조회 자료의 구체적 사실(날짜·이벤트·수치 등)을 근거로 설명해야 한다. 원인의 개수는 요구하지 않는다. 기사 제목이나 매체명까지는 요구하지 않는다. 근거를 찾지 못한 부분은 못 찾았다고 밝히면 되며, 근거 없는 원인을 사실처럼 추가하면 fail. 원인을 전혀 설명하지 못하고 되묻거나 수치만 나열하면 fail.",
        "company": "삼성전자",
        "expected_sources": {"dart", "news"},
        "numeric_target": {
            "company": "삼성전자",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "영업이익",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "q6_multihop_skhynix_categories",
        "question": "SK하이닉스 2026년 상반기 누적 실적 원인을 회사 내부 이슈, 산업 동향, 거시경제 이슈로 나눠서 설명해줘",
        # 예전 기준("세 카테고리 모두 근거 있는 원인")은 근거가 없는 카테고리를 일반론으로 채우게 만들었다
        # → 근거 없는 카테고리는 솔직히 밝히면 통과, 지어내면 fail (2026-09-24 수정)
        "rubric": "원인을 ① 회사 내부 ② 산업 동향 ③ 거시경제 세 카테고리로 나눠 정리해야 한다. 각 카테고리에서 조회 자료의 구체적 사실(날짜·이벤트·수치 등)로 뒷받침되는 원인을 제시하되, 근거를 찾지 못한 카테고리는 '근거를 찾지 못했다'고 명시하면 된다. 기사 제목이나 매체명까지는 요구하지 않는다. 근거 없는 원인이나 일반론을 사실처럼 채워 넣으면 fail. 세 카테고리 중 근거 있는 원인이 하나도 없으면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"dart", "news"},
        "numeric_target": {
            "company": "SK하이닉스",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "영업이익",
            "amount_field": "thstrm_add_amount",
        },
    },
    # --- 주가/밸류에이션 (get_stock_price 도구 전용) ---
    {
        "id": "q7_price_return",
        "question": "삼성전자 2026년 9월 1일부터 9월 23일까지 주가 수익률이 몇 %야?",
        "rubric": "해당 기간 주가 수익률을 %로 제시해야 한다 (참고 정답과 ±0.5%p 이내). 다른 기간의 수익률을 제시하거나 수치가 없으면 fail.",
        "company": "삼성전자",
        "expected_sources": {"price"},
        "numeric_target": {
            "kind": "price_return",
            "company": "삼성전자",
            "start": "2026-09-01",
            "end": "2026-09-23",
        },
    },
    {
        "id": "q8_per",
        # KRX 로그인(.env의 KRX_ID/KRX_PW)이 없으면 이 질문은 numeric 채점이 생략됨(None) —
        # 인용 출처(price 도구 호출 여부) 채점은 로그인과 무관하게 정상 작동.
        "question": "SK하이닉스 2026년 9월 23일 기준 PER가 몇 배야?",
        "rubric": "2026-09-23 기준 PER을 배수로 제시해야 한다. 수치가 없거나 참고 정답과 다르면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"price"},
        "numeric_target": {"kind": "per", "company": "SK하이닉스", "on": "2026-09-23"},
    },
    # --- 재무 비율 / 기업 비교 (DART 도구, 단일 기업 내 계산 vs 여러 기업 조회) ---
    {
        "id": "q9_operating_margin",
        "question": "삼성전자 2026년 상반기 누적 영업이익률(영업이익/매출액)이 몇 %야?",
        "rubric": "2026년 상반기 누적 영업이익률을 %로 제시해야 한다 (참고 정답과 ±0.5%p 이내). 수치가 없거나 분기 단독 기준으로 계산하면 fail.",
        "company": "삼성전자",
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "ratio",
            "company": "삼성전자",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "numerator_account": "영업이익",
            "denominator_account": "매출액",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "q10_compare_margin",
        "question": "삼성전자와 SK하이닉스 중 2026년 상반기 누적 영업이익률이 더 높은 곳은 어디야? 두 회사 수치를 각각 알려줘",
        "rubric": "두 회사의 상반기 누적 영업이익률을 각각 제시하고, 어느 회사가 더 높은지 올바르게 결론 내려야 한다. 한 회사 수치가 빠지거나, 더 높은 회사를 틀리게 말하거나, 답변 안에서 더 높은 회사를 서로 다르게 말하면(자기모순) fail.",
        "company": "삼성전자",  # 대표 company (baseline이 단일 기업 도구 호출에 쓸 값)
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "ratio_pair",
            "company_a": {
                "company": "삼성전자",
                "bsns_year": "2026",
                "reprt_code": "11012",
                "numerator_account": "영업이익",
                "denominator_account": "매출액",
                "amount_field": "thstrm_add_amount",
            },
            "company_b": {
                "company": "SK하이닉스",
                "bsns_year": "2026",
                "reprt_code": "11012",
                "numerator_account": "영업이익",
                "denominator_account": "매출액",
                "amount_field": "thstrm_add_amount",
            },
        },
    },
    # --- 잘못된 전제 (실제로는 반대 방향인데 그렇다고 단정하고 묻는 질문) ---
    {
        "id": "q11_false_premise_price_drop",
        # 실측: 삼성전자 종가 2026-09-01 261,000원 -> 2026-09-03 250,000원 (-4.2%, 하락).
        # "올랐어?"라고 잘못 전제하고 물었을 때 이걸 바로잡는지 채점.
        "question": "삼성전자 주가가 2026년 9월 1일부터 9월 3일까지 왜 올랐어?",
        "rubric": "질문의 '올랐다'는 전제가 사실과 다르다는 것(실제로는 하락)을 실제 수치와 함께 명확히 지적해야 한다. 전제를 바로잡았다면 상승 원인을 설명하지 못한 것은 fail 사유가 아니다(실제로 오르지 않았으므로). 상승을 사실로 받아들여 상승 원인을 설명하거나, 전제를 확인하지 못했다고만 하면 fail.",
        "company": "삼성전자",
        "expected_sources": {"price"},
        "numeric_target": {
            "kind": "price_direction",
            "company": "삼성전자",
            "start": "2026-09-01",
            "end": "2026-09-03",
        },
    },
]

# 결과를 보기 전에 확정 (2026-09-24). 튜닝 셋과 유형이 겹치지 않게: 재무상태표 비율, 성장률, 다른 기간 수익률,
# PBR, 재무 데이터로 잘못된 전제 교정, 주가+재무+뉴스 3도구 멀티홉.
HELDOUT_QUESTIONS = [
    {
        "id": "h1_debt_ratio",
        "question": "SK하이닉스 2026년 6월 말 기준 부채비율(부채총계÷자본총계)은 몇 %야?",
        "rubric": "2026년 6월 말 기준 부채비율을 %로 제시해야 한다 (참고 정답과 ±0.5%p 이내). 수치가 없거나 다른 시점·다른 계정으로 계산하면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "ratio",
            "company": "SK하이닉스",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "numerator_account": "부채총계",
            "denominator_account": "자본총계",
            "amount_field": "thstrm_amount",  # 재무상태표는 시점 값이라 누적 필드가 없음
        },
    },
    {
        "id": "h2_revenue_growth",
        "question": "삼성전자 2026년 상반기 누적 매출액은 전년 동기보다 몇 % 늘었어?",
        "rubric": "2026년 상반기 누적 매출액의 전년 동기 대비 증가율을 %로 제시해야 한다. 수치가 없거나 분기 단독 기준으로 계산하면 fail.",
        "company": "삼성전자",
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "growth",
            "company": "삼성전자",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "매출액",
        },
    },
    {
        "id": "h3_price_return_aug",
        "question": "삼성전자 주가는 2026년 8월 3일부터 8월 31일까지 몇 % 변했어?",
        "rubric": "해당 기간 주가 변화율을 %로 제시해야 한다 (참고 정답과 ±0.5%p 이내). 다른 기간의 수익률을 제시하거나 수치가 없으면 fail.",
        "company": "삼성전자",
        "expected_sources": {"price"},
        "numeric_target": {"kind": "price_return", "company": "삼성전자", "start": "2026-08-03", "end": "2026-08-31"},
    },
    {
        "id": "h4_pbr",
        "question": "삼성전자 2026년 9월 23일 기준 PBR은 몇 배야?",
        "rubric": "2026-09-23 기준 PBR을 배수로 제시해야 한다. 수치가 없거나 참고 정답과 다르면 fail.",
        "company": "삼성전자",
        "expected_sources": {"price"},
        "numeric_target": {"kind": "per", "field": "PBR", "company": "삼성전자", "on": "2026-09-23"},
    },
    {
        "id": "h5_false_premise_dart",
        "question": "SK하이닉스 2026년 상반기 누적 영업이익이 전년 동기보다 왜 줄었어?",
        "rubric": "질문의 '줄었다'는 전제가 사실과 다르다는 것(실제로는 증가)을 전년 동기와 당기 수치로 명확히 지적해야 한다. 전제를 바로잡았다면 감소 원인을 설명하지 못한 것은 fail 사유가 아니다. 감소를 사실로 받아들여 감소 원인을 설명하면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "growth",
            "company": "SK하이닉스",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "영업이익",
        },
    },
    {
        "id": "h6_multihop_price_earnings",
        "question": "SK하이닉스 주가가 2026년 9월 1일부터 23일까지 오른 게 실적 때문이야? 주가 변화, 상반기 실적 숫자, 관련 뉴스를 근거로 설명해줘",
        "rubric": "① 해당 기간 주가 변화율 수치 ② 상반기 실적 수치(매출·영업이익 등) ③ 관련 뉴스의 구체적 사실(날짜·이벤트·수치 등)을 모두 제시하고, 주가 상승과 실적의 관계를 근거 수준에 맞게 설명해야 한다(근거 없이 인과를 단정하면 안 됨). 셋 중 하나라도 빠지면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"price", "dart", "news"},
        "numeric_target": {"kind": "price_return", "company": "SK하이닉스", "start": "2026-09-01", "end": "2026-09-23"},
    },
]


# 새 held-out (v2). 2026-09-24, v1 held-out을 보고 에이전트를 고친 뒤 결과를 보기 전에 확정.
# v1이 오염됐기 때문에 만든 셋이라, 튜닝 셋·v1과 유형이 겹치지 않게 골랐다: 연간/3분기 보고서 기간 선택,
# 순이익률, "더 낮은 쪽" 기업 비교(근소한 차이), 반대 방향 주가 전제, 밸류에이션 판단.
# 규칙은 v1과 같다: 이 결과를 보고 에이전트 프롬프트를 고치지 말 것 (버그 수정은 예외).
HELDOUT_V2_QUESTIONS = [
    {
        "id": "v2_1_annual_revenue",
        "question": "삼성전자 2025년 연간 매출액은 얼마야?",
        "rubric": "2025년 연간(사업보고서 기준) 매출액을 구체적 금액으로 제시해야 한다. 2026년 상반기 등 다른 기간의 값을 제시하거나 수치가 없으면 fail.",
        "company": "삼성전자",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "삼성전자",
            "bsns_year": "2025",
            "reprt_code": "11011",
            "account_name": "매출액",
            "amount_field": "thstrm_amount",
        },
    },
    {
        "id": "v2_2_q3_cumulative_op",
        "question": "SK하이닉스 2025년 1~9월 누적 영업이익은 얼마야?",
        "rubric": "2025년 1~9월(3분기 누적) 영업이익을 구체적 금액으로 제시해야 한다. 3분기 단독 값이나 다른 기간의 값을 결론으로 제시하거나 수치가 없으면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "SK하이닉스",
            "bsns_year": "2025",
            "reprt_code": "11014",
            "account_name": "영업이익",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "v2_3_net_margin",
        "question": "삼성전자 2026년 상반기 누적 순이익률(당기순이익÷매출액)은 몇 %야?",
        "rubric": "2026년 상반기 누적 기준 순이익률을 %로 제시해야 한다 (참고 정답과 ±0.5%p 이내). 영업이익률 등 다른 비율을 제시하거나 수치가 없으면 fail.",
        "company": "삼성전자",
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "ratio",
            "company": "삼성전자",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "numerator_account": "당기순이익(손실)",
            "denominator_account": "매출액",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "v2_4_compare_debt_lower",
        "question": "삼성전자와 SK하이닉스 중 2026년 6월 말 기준 부채비율(부채총계÷자본총계)이 더 낮은 곳은 어디야? 두 회사 수치도 알려줘",
        "rubric": "두 회사의 2026년 6월 말 부채비율을 각각 제시하고, 더 낮은 회사를 올바르게 결론 내려야 한다. 한 회사 수치가 빠지거나, 더 낮은 회사를 틀리게 말하거나, 답변 안에서 결론이 모순되면 fail.",
        "company": "삼성전자",
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "ratio_pair",
            "company_a": {
                "company": "삼성전자",
                "bsns_year": "2026",
                "reprt_code": "11012",
                "numerator_account": "부채총계",
                "denominator_account": "자본총계",
                "amount_field": "thstrm_amount",
            },
            "company_b": {
                "company": "SK하이닉스",
                "bsns_year": "2026",
                "reprt_code": "11012",
                "numerator_account": "부채총계",
                "denominator_account": "자본총계",
                "amount_field": "thstrm_amount",
            },
        },
    },
    {
        "id": "v2_5_false_premise_price_up",
        "question": "SK하이닉스 주가가 2026년 8월 3일부터 8월 31일까지 왜 떨어졌어?",
        "rubric": "질문의 '떨어졌다'는 전제가 사실과 다르다는 것(실제로는 상승)을 실제 수치와 함께 명확히 지적해야 한다. 전제를 바로잡았다면 하락 원인을 설명하지 못한 것은 fail 사유가 아니다. 하락을 사실로 받아들여 하락 원인을 설명하거나, 전제를 확인하지 못했다고만 하면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"price"},
        "numeric_target": {"kind": "price_direction", "company": "SK하이닉스", "start": "2026-08-03", "end": "2026-08-31"},
    },
    {
        "id": "v2_6_valuation_judgment",
        "question": "SK하이닉스는 2026년 9월 23일 기준 PER이 몇 배야? 상반기 실적이 늘어난 속도를 감안하면 비싼 편이야, 싼 편이야? 근거를 들어 설명해줘",
        "rubric": "① 2026-09-23 기준 PER 수치 ② 상반기 실적의 전년 동기 대비 증가(수치)를 제시하고, ③ 둘을 연결해 비싼지 싼지에 대한 판단 근거를 설명해야 한다. 판단은 근거 수준에 맞게 해야 하며, 조회 자료에 없는 전망이나 목표주가를 사실처럼 단정하면 fail. ①~③ 중 하나라도 빠지면 fail.",
        "company": "SK하이닉스",
        "expected_sources": {"price", "dart"},
        "numeric_target": {"kind": "per", "field": "PER", "company": "SK하이닉스", "on": "2026-09-23"},
    },
]


# held-out v3. 2026-09-26, 지원 종목 확장(요청 시 DART 수집, 별칭, 핵심 종목군 50개사, 업종 비교 도구)과
# 최종 답변 모델 분리(mini) 이후, 결과를 보기 전에 확정. 새 기능을 직접 묻는 질문으로 구성:
# DB에 없던 회사(크래프톤), 별칭(네이버→NAVER), 업종 1위, 경쟁사 대비 위치, 새 회사의 과거 연도, 업종 비교+뉴스 멀티홉.
# 규칙은 같다: 이 결과를 보고 에이전트 프롬프트를 고치지 말 것 (버그 수정은 예외).
HELDOUT_V3_QUESTIONS = [
    {
        "id": "v3_1_on_demand_company",
        # 크래프톤은 이 문항을 만들 때 로컬 DB에 없었음 → 도구가 DART에서 즉석으로 받아와야 풀 수 있다
        "question": "크래프톤 2026년 상반기 누적 영업이익은 얼마야?",
        "rubric": "2026년 상반기(1~6월) 누적 영업이익을 구체적 금액으로 제시해야 한다. 분기 단독 금액 등 다른 값을 결론으로 제시하거나, 데이터가 없다며 답하지 못하면 fail.",
        "company": "크래프톤",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "크래프톤",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "영업이익",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "v3_2_alias_company",
        "question": "네이버 2026년 상반기 누적 매출액은 얼마야?",
        "rubric": "NAVER의 2026년 상반기(1~6월) 누적 매출액을 구체적 금액으로 제시해야 한다. 회사를 찾지 못했다며 답하지 못하거나 다른 값을 제시하면 fail.",
        "company": "네이버",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "NAVER",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "매출액",
            "amount_field": "thstrm_add_amount",
        },
    },
    {
        "id": "v3_3_peer_top",
        "question": "반도체 소재 업체들 중 2026년 상반기 누적 영업이익률이 가장 높은 곳은 어디고 몇 %야?",
        "rubric": "반도체 소재 업종에서 영업이익률 1위 회사와 그 수치를 제시해야 한다. 1위 회사를 틀리게 말하거나 수치가 없으면 fail.",
        "company": "솔브레인",  # 고정 파이프라인이 쓸 대표 회사 (소재 업종)
        "expected_sources": {"dart"},
        "numeric_target": {"kind": "peer_top", "group": "소재", "metric": "영업이익률"},
    },
    {
        "id": "v3_4_peer_position",
        "question": "현대자동차의 2026년 상반기 누적 매출액 증가율은 같은 업종 경쟁사들과 비교하면 어느 수준이야?",
        "rubric": "현대자동차의 상반기 누적 매출액 전년 동기 대비 증가율을 제시하고, 같은 업종 경쟁사(예: 기아, 현대모비스)의 증가율과 비교해 상대적 위치를 올바르게 결론 내려야 한다. 경쟁사 수치 없이 결론만 내리거나 위치를 틀리게 말하면 fail.",
        "company": "현대자동차",
        "expected_sources": {"dart"},
        "numeric_target": {
            "kind": "growth",
            "company": "현대자동차",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "account_name": "매출액",
        },
    },
    {
        "id": "v3_5_new_company_past_year",
        "question": "LG전자 2025년 연간 영업이익은 얼마였고, 전년보다 늘었어 줄었어?",
        "rubric": "LG전자의 2025년 연간 영업이익을 구체적 금액으로 제시하고, 전년(2024년) 대비 증감 방향을 수치와 함께 올바르게 말해야 한다. 다른 기간 값을 제시하거나 방향을 틀리면 fail.",
        "company": "LG전자",
        "expected_sources": {"dart"},
        "numeric_target": {
            "company": "LG전자",
            "bsns_year": "2025",
            "reprt_code": "11011",
            "account_name": "영업이익",
            "amount_field": "thstrm_amount",
        },
    },
    {
        "id": "v3_6_multihop_peer_news",
        "question": "한미반도체 영업이익률이 후공정 장비 경쟁사보다 높은 이유가 뭐야? 수치와 뉴스 근거로 설명해줘",
        "rubric": "① 한미반도체의 영업이익률 수치 ② 후공정 장비 경쟁사(예: 테크윙)의 영업이익률과의 비교 ③ 그 차이의 원인을 조회 자료의 구체적 사실로 설명해야 한다. 원인 근거를 찾지 못한 부분은 솔직히 밝히면 되며, 근거 없는 원인을 사실처럼 단정하면 fail. ①②가 빠져도 fail.",
        "company": "한미반도체",
        "expected_sources": {"dart", "news"},
        "numeric_target": {
            "kind": "ratio",
            "company": "한미반도체",
            "bsns_year": "2026",
            "reprt_code": "11012",
            "numerator_account": "영업이익",
            "denominator_account": "매출액",
            "amount_field": "thstrm_add_amount",
        },
    },
]

# 결과를 보기 전에 고정한 해시. heldout_v4.json을 바꾸면 여기서 멈춘다 — 바꿔야 한다면 v5를 새로 만들 것.
HELDOUT_V4_SHA256 = "cc3a9a7cda06e0da24f868fa93aa434ed03b02ddac3622623ab3421115364a6d"
_V4_PATH = Path(__file__).with_name("heldout_v4.json")


def _load_heldout_v4() -> list[dict]:
    raw = _V4_PATH.read_bytes().replace(b"\r\n", b"\n")  # Windows git이 줄바꿈을 바꿔도 같은 해시가 나오게
    digest = hashlib.sha256(raw).hexdigest()
    if digest != HELDOUT_V4_SHA256:
        raise RuntimeError(f"heldout_v4.json이 고정된 뒤 바뀌었습니다 (해시 {digest[:12]}…). held-out을 고치지 말고 새로 만드세요.")
    questions = json.loads(raw)["questions"]
    for q in questions:
        q["expected_sources"] = set(q["expected_sources"])
    return questions


HELDOUT_V4_QUESTIONS = _load_heldout_v4()

for _q in TUNING_QUESTIONS:
    _q["split"] = "tuning"
for _q in HELDOUT_QUESTIONS:
    _q["split"] = "heldout"
for _q in HELDOUT_V2_QUESTIONS:
    _q["split"] = "heldout_v2"
for _q in HELDOUT_V3_QUESTIONS:
    _q["split"] = "heldout_v3"
for _q in HELDOUT_V4_QUESTIONS:
    _q["split"] = "heldout_v4"

GOLD_QUESTIONS = TUNING_QUESTIONS + HELDOUT_QUESTIONS + HELDOUT_V2_QUESTIONS + HELDOUT_V3_QUESTIONS + HELDOUT_V4_QUESTIONS
