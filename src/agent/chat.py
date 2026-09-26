"""터미널 대화 모드: python -m src.agent.chat [--session 이름]

한 프로세스를 계속 띄워 두므로 임베딩 모델(bge-m3)·기업 목록을 한 번만 불러온다. 임베딩 모델은 시작하자마자
백그라운드 스레드에서 미리 불러와, 사용자가 첫 질문을 입력하는 동안 로딩이 끝나게 한다 (첫 뉴스 검색 대기 단축).
같은 --session 이름으로 다시 켜면 이전 대화를 이어간다.
"""

import argparse
import threading
from datetime import datetime

from ..data import news_client as nc
from .memory import ask, load_history


def _warm_up() -> None:
    try:
        nc._get_embedding_model()
    except Exception:  # noqa: BLE001 — 미리 불러오기 실패는 첫 검색 때 다시 시도되므로 무시
        pass


def _print_result(result: dict) -> None:
    if result["standalone_question"]:
        print(f"  (이해한 질문: {result['standalone_question']})")
    for t in result["transcript"]:
        print(f"  · {t['tool']}({t['args']}) {t['elapsed']}초")
    print(f"\n{result['answer']}\n")
    tokens = ", ".join(f"{m} {u['calls']}회 입력 {u['prompt']:,}/출력 {u['completion']:,}" for m, u in result["usage"].items())
    print(f"  [토큰] {tokens}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=datetime.now().strftime("chat-%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    threading.Thread(target=_warm_up, daemon=True).start()
    history = load_history(args.session, limit=100)
    print(f"세션: {args.session}" + (f" (이전 대화 {len(history)}개 이어감)" if history else "") + " · 종료: 빈 줄 또는 Ctrl+C")
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        _print_result(ask(args.session, question))


if __name__ == "__main__":
    main()
