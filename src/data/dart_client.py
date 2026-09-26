"""DART Open API 클라이언트.

Phase 1: 기업개황 / 재무제표(XBRL) / 공시 원문 조회.
API 신청: https://opendart.fss.or.kr (회원가입 → 인증키 신청/관리)

DART의 모든 조회 API는 종목코드(예: 005930)가 아니라 DART 고유의 8자리 corp_code를
요구한다. corp_code는 전체 기업 목록(zip)을 받아야 알 수 있어 get_corp_code()에서
로컬에 캐싱해 처리한다. 응답 구조는 Notion "DART API 데이터 구조 정리 (Phase 1)" 참고.
"""

import difflib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv

load_dotenv()  # .env 파일을 명시적으로 로드하지 않으면 셸에 키를 직접 export한 세션에서만 동작함

DART_API_KEY = os.getenv("DART_API_KEY")
BASE_URL = "https://opendart.fss.or.kr/api"

_CORP_CODE_CACHE = Path(__file__).parent / ".dart_corp_code_cache.json"


class DartApiError(RuntimeError):
    """DART API가 status != '000'으로 에러를 응답했을 때 발생.

    status="013"(조회된 데이터가 없습니다)은 아직 공시가 안 올라온 정상적인 상황이므로
    호출하는 쪽에서 이 코드를 보고 진짜 에러와 구분해 건너뛸 수 있다.
    """

    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(f"DART API 에러 [{status}]: {message}")


def _check_status(data: dict) -> dict:
    if data.get("status") != "000":
        raise DartApiError(data.get("status"), data.get("message"))
    return data


def _get_json(url: str, params: dict, retries: int = 3, backoff: float = 1.0) -> dict:
    """DART가 가끔(랜덤) JSON 대신 HTML 에러 페이지를 200으로 반환하는 경우가 있어 재시도한다."""
    last_error: Exception | None = None
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        try:
            return resp.json()
        except requests.exceptions.JSONDecodeError as e:
            last_error = e
            time.sleep(backoff * (attempt + 1))
    raise DartApiError("JSON_DECODE_FAILED", f"{retries}번 재시도에도 유효한 JSON을 반환하지 않았습니다: {last_error}")


def _download_corp_code_index() -> dict:
    """전체 기업 목록(zip)을 받아 stock_code/corp_name -> corp_code 매핑으로 변환."""
    resp = requests.get(f"{BASE_URL}/corpCode.xml", params={"crtfc_key": DART_API_KEY}, timeout=30)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    xml_bytes = zf.read(zf.namelist()[0])
    root = ElementTree.fromstring(xml_bytes)

    by_stock_code = {}
    by_name = {}
    for item in root.iter("list"):
        corp_code = item.findtext("corp_code")
        corp_name = item.findtext("corp_name")
        stock_code = (item.findtext("stock_code") or "").strip()
        entry = {"corp_code": corp_code, "corp_name": corp_name, "stock_code": stock_code}
        if stock_code:
            by_stock_code[stock_code] = entry
        by_name[corp_name] = entry

    return {"by_stock_code": by_stock_code, "by_name": by_name}


_index_in_memory: dict | None = None  # 30MB 파일을 호출마다 다시 읽지 않도록 프로세스 안에서 한 번만 읽는다


def _load_corp_code_index(force_refresh: bool = False) -> dict:
    global _index_in_memory
    if not force_refresh and _index_in_memory is not None:
        return _index_in_memory
    if not force_refresh and _CORP_CODE_CACHE.exists():
        _index_in_memory = json.loads(_CORP_CODE_CACHE.read_text(encoding="utf-8"))
        return _index_in_memory

    _index_in_memory = _download_corp_code_index()
    _CORP_CODE_CACHE.write_text(json.dumps(_index_in_memory, ensure_ascii=False), encoding="utf-8")
    return _index_in_memory


# DART 공식 이름과 흔히 부르는 이름이 다른 경우 (공식 이름이 영문이거나 줄임말로 부를 때)
_ALIASES = {"네이버": "NAVER", "현대차": "현대자동차", "기아차": "기아", "엘지전자": "LG전자", "포스코": "POSCO홀딩스"}


def _find_entry(name_or_stock_code: str, force_refresh: bool = False) -> dict:
    index = _load_corp_code_index(force_refresh=force_refresh)
    name_or_stock_code = _ALIASES.get(name_or_stock_code.strip(), name_or_stock_code.strip())
    entry = index["by_stock_code"].get(name_or_stock_code) or index["by_name"].get(name_or_stock_code)
    if entry is None:
        raise KeyError(f"'{name_or_stock_code}'에 해당하는 기업을 찾지 못했습니다.")
    return entry


def get_corp_code(name_or_stock_code: str, force_refresh: bool = False) -> str:
    """회사명 또는 종목코드로 DART corp_code를 찾는다.

    최초 호출 시 전체 기업 목록(zip, 약 30MB)을 받아 로컬(.dart_corp_code_cache.json)에
    캐싱하고, 이후 호출은 캐시를 재사용한다. 이미 8자리 corp_code를 받으면 그대로 돌려준다.
    """
    value = name_or_stock_code.strip()
    if len(value) == 8 and value.isdigit():  # 종목코드는 6자리라 8자리 숫자면 corp_code
        return value
    return _find_entry(value, force_refresh)["corp_code"]


def suggest_listed_names(query: str, n: int = 5) -> list[str]:
    """상장사 중 이름이 비슷한 후보. 이름이 정확히 일치하지 않을 때(예: '현대차' → '현대자동차') 안내용."""
    names = [e["corp_name"] for e in _load_corp_code_index()["by_stock_code"].values()]
    containing = sorted((nm for nm in names if query in nm or nm in query), key=len)
    similar = difflib.get_close_matches(query, names, n=n, cutoff=0.4)
    return list(dict.fromkeys(containing + similar))[:n]


def get_stock_code(name_or_stock_code: str) -> str:
    """회사명 또는 종목코드로 6자리 종목코드(주가 조회용)를 찾는다. 비상장사면 KeyError."""
    stock_code = _find_entry(name_or_stock_code)["stock_code"]
    if not stock_code:
        raise KeyError(f"'{name_or_stock_code}'은(는) 상장 종목코드가 없습니다 (비상장사).")
    return stock_code


_OVERVIEW_TTL_SECONDS = 30 * 24 * 3600  # 회사명·결산월·업종 코드는 거의 안 바뀜


def get_company_overview(corp_code: str) -> dict:
    """기업개황 조회. 재무 도구가 부를 때마다 DART에 다시 묻던 것을 캐시(30일)로 바꿨다."""
    from . import cache  # dart_client만 쓰는 스크립트가 numpy 등을 불러오지 않도록 필요할 때 import

    key = cache.make_key(corp_code)
    cached = cache.get("dart_overview", key)
    if cached is not None:
        return cached
    data = _check_status(_get_json(f"{BASE_URL}/company.json", {"crtfc_key": DART_API_KEY, "corp_code": corp_code}))
    cache.put("dart_overview", key, data, _OVERVIEW_TTL_SECONDS)
    return data


def _parse_amount(raw: str | None) -> int | None:
    """금액 문자열("1,234" 또는 음수 "-1,234")을 정수로. 값이 없으면 None.

    일부 회사(보험사 등)는 빈 금액을 "-"로 보내서, 예전 코드는 int("-")에서 수집이 멈췄다.
    """
    if not raw:
        return None
    cleaned = raw.replace(",", "").strip()
    if cleaned in ("", "-"):
        return None
    return int(cleaned)


def get_financial_statement(corp_code: str, year: str, report_code: str = "11011") -> list[dict]:
    """재무제표(XBRL) 조회.

    report_code 기본값 11011 = 사업보고서(연간). 한 번의 호출로 당기/전기/전전기
    3개년 금액을 계정과목별로 받는다. 금액 문자열(콤마 포함)은 정수로 파싱해서 반환.
    """
    data = _get_json(
        f"{BASE_URL}/fnlttSinglAcnt.json",
        {
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bsns_year": year,
            "reprt_code": report_code,
        },
    )
    data = _check_status(data)

    return [
        {
            "fs_div": row["fs_div"],  # CFS=연결재무제표, OFS=별도재무제표 — 같은 계정명이 두 번씩 나오므로 구분 필수
            "fs_name": row["fs_nm"],
            "sj_name": row["sj_nm"],
            "account_name": row["account_nm"],
            "thstrm_amount": _parse_amount(row.get("thstrm_amount")),
            "frmtrm_amount": _parse_amount(row.get("frmtrm_amount")),
            "bfefrmtrm_amount": _parse_amount(row.get("bfefrmtrm_amount")),
            # 반기/3분기보고서의 손익계산서 항목에만 존재. thstrm_amount는 "해당 분기 단독" 값이고
            # thstrm_add_amount는 "연초~해당 분기까지 누적" 값 — 재무상태표 항목/1분기·연간보고서에는 없음(None).
            "thstrm_add_amount": _parse_amount(row.get("thstrm_add_amount")),
            "frmtrm_add_amount": _parse_amount(row.get("frmtrm_add_amount")),
        }
        for row in data["list"]
    ]


# TODO: 이후 Phase에서 구현
# - search_disclosures(corp_code, start_date, end_date)
