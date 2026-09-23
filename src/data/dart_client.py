"""DART Open API 클라이언트.

Phase 1: 기업개황 / 재무제표(XBRL) / 공시 원문 조회.
API 신청: https://opendart.fss.or.kr (회원가입 → 인증키 신청/관리)

DART의 모든 조회 API는 종목코드(예: 005930)가 아니라 DART 고유의 8자리 corp_code를
요구한다. corp_code는 전체 기업 목록(zip)을 받아야 알 수 있어 get_corp_code()에서
로컬에 캐싱해 처리한다. 응답 구조는 Notion "DART API 데이터 구조 정리 (Phase 1)" 참고.
"""

import io
import json
import os
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import requests

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
    raise DartApiError(f"DART API가 {retries}번 재시도에도 유효한 JSON을 반환하지 않았습니다: {last_error}")


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


def _load_corp_code_index(force_refresh: bool = False) -> dict:
    if not force_refresh and _CORP_CODE_CACHE.exists():
        return json.loads(_CORP_CODE_CACHE.read_text(encoding="utf-8"))

    index = _download_corp_code_index()
    _CORP_CODE_CACHE.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return index


def get_corp_code(name_or_stock_code: str, force_refresh: bool = False) -> str:
    """회사명 또는 종목코드로 DART corp_code를 찾는다.

    최초 호출 시 전체 기업 목록(zip, 약 30MB)을 받아 로컬(.dart_corp_code_cache.json)에
    캐싱하고, 이후 호출은 캐시를 재사용한다.
    """
    index = _load_corp_code_index(force_refresh=force_refresh)

    entry = index["by_stock_code"].get(name_or_stock_code) or index["by_name"].get(name_or_stock_code)
    if entry is None:
        raise KeyError(f"'{name_or_stock_code}'에 해당하는 기업을 찾지 못했습니다.")
    return entry["corp_code"]


def get_company_overview(corp_code: str) -> dict:
    """기업개황 조회."""
    data = _get_json(f"{BASE_URL}/company.json", {"crtfc_key": DART_API_KEY, "corp_code": corp_code})
    return _check_status(data)


def _parse_amount(raw: str | None) -> int | None:
    if not raw:
        return None
    return int(raw.replace(",", ""))


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
