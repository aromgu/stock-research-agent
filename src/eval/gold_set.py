"""Phase 5: 채점용 골드 질문셋.

각 질문에 "이 출처가 인용돼야 한다"는 정답 라벨(expected_sources)과,
숫자로 채점 가능한 질문이면 DART DB에서 정답값을 찾기 위한 좌표(numeric_target)를 붙인다.
숫자 정답 자체는 하드코딩하지 않고 채점 시점에 DB에서 직접 조회한다 (데이터가 바뀌어도 안 썩게).
"""

GOLD_QUESTIONS = [
    {
        "id": "q1_dart_revenue",
        # "반기보고서 기준 매출액"은 모호함(해당 분기 단독 171조 vs 상반기 누적 305조 둘 다
        # 말이 됨) — LLM이 어느 쪽으로 답해도 자연스러워서 채점이 운에 좌우됐다.
        # "1~6월 누적"으로 못박아 thstrm_add_amount와 짝을 맞춘다.
        "question": "삼성전자 2026년 1~6월(상반기) 누적 매출액이 얼마야?",
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
        "company": "삼성전자",
        "expected_sources": {"price", "news"},
        "numeric_target": None,  # "최근"이 모호해서 자동 채점 가능한 고정 구간이 아님
    },
    {
        "id": "q5_multihop_samsung",
        "question": "삼성전자 2026년 상반기 누적 실적이 왜 이렇게 나왔는지 영업이익 숫자랑 같이 설명해줘",
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
        "company": "SK하이닉스",
        "expected_sources": {"price"},
        "numeric_target": {"kind": "per", "company": "SK하이닉스", "on": "2026-09-23"},
    },
    # --- 재무 비율 / 기업 비교 (DART 도구, 단일 기업 내 계산 vs 여러 기업 조회) ---
    {
        "id": "q9_operating_margin",
        "question": "삼성전자 2026년 상반기 누적 영업이익률(영업이익/매출액)이 몇 %야?",
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
