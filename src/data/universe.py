"""미리 데이터를 수집해두는 핵심 종목군 (반도체 밸류체인 + 코스피 시가총액 상위).

그 밖의 회사는 도구가 요청받을 때 DART에서 받아온다 (tools._ensure_reports). 미리 모아두는 이유는
"장비주 중 영업이익률 1위" 같은 업종 비교 질문에 답하려면 비교 대상 전체의 데이터가 있어야 하기 때문.

실제 목록은 build_universe.py가 만든 data/universe_snapshot.json을 읽는다. 시가총액 순위는 날마다
바뀌므로 특정 날짜 기준으로 고정한 스냅샷을 써야 평가를 재현할 수 있다.
"""

import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "universe_snapshot.json"

VALUE_CHAIN_LABEL = "반도체 밸류체인"
KOSPI_TOP_LABEL = "코스피 시가총액 상위"

# 반도체 밸류체인 세부 업종. 이 프로젝트용으로 단순화한 분류다 (여러 사업을 하는 회사는 주력 하나로만 분류).
SEMICONDUCTOR_SEGMENTS = {
    "메모리": ["삼성전자", "SK하이닉스"],
    "파운드리": ["DB하이텍"],
    "전공정 장비": ["원익IPS", "주성엔지니어링", "유진테크", "피에스케이", "테스", "HPSP", "이오테크닉스"],
    "후공정 장비": ["한미반도체", "테크윙"],
    "소재": ["솔브레인", "동진쎄미켐", "원익머트리얼즈", "후성", "티씨케이"],
    "부품·테스트": ["리노공업", "ISC"],
    "패키징·기판": ["하나마이크론", "네패스", "심텍"],
}


def load_universe() -> list[dict]:
    """스냅샷의 회사 목록. 각 항목: name, stock_code, corp_code, induty_code, segment(반도체면), groups."""
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError("종목군 스냅샷이 없습니다. 먼저 `python -m src.data.build_universe`를 실행하세요.")
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["companies"]


def snapshot_date() -> str:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["as_of"]
