"""대화 메모리: 이어지는 질문("그럼 SK하이닉스는?")을 앞 대화를 보고 혼자서도 뜻이 통하는 질문으로 바꾼다.

이전 대화 전체를 플래너에 계속 붙이지 않고, 질문 하나만 다시 쓰는 방식을 택했다:
- 플래너 입력이 대화가 길어져도 늘지 않아 토큰이 일정하다 (도구 결과까지 붙이면 턴마다 수천 토큰씩 늘어남).
- 이전 턴에서 조회한 재무·주가·뉴스는 도구 캐시에 남아 있어, 같은 조회를 다시 해도 API를 부르지 않는다.
- 재작성된 질문이 로그에 남아 "에이전트가 무엇을 물은 것으로 이해했는지" 확인할 수 있다.

대화는 data/cache.db에 저장되어 프로그램을 다시 켜도 같은 세션 이름으로 이어갈 수 있다.
"""

import sqlite3
import time

from ..data import cache
from .planner import _get_client, record_usage, run_agent
from .prompts import PLANNER_MODEL

_HISTORY_TURNS = 3  # 재작성에 참고할 최근 대화 수
_ANSWER_CHARS = 600  # 이전 답변은 앞부분만 참고 ("아까 두 번째 원인" 같은 지칭을 풀 정도면 충분)

_REWRITE_PROMPT = (
    "아래는 종목 리서치 어시스턴트와 사용자의 이전 대화와, 사용자의 새 질문이다.\n"
    "새 질문을 이전 대화 없이도 뜻이 통하는 하나의 질문으로 다시 써.\n"
    "- '그럼 ~는?', '그 회사', '아까 말한 원인', '같은 기간' 같은 지칭은 이전 대화에서 가리키는 회사명·기간·지표·내용으로 바꿔.\n"
    "- 새 질문이 이미 혼자서 뜻이 통하면(다른 회사·주제로 바뀐 경우 포함) 그대로 출력해.\n"
    "- 질문에 없는 조건을 새로 만들지 마. 설명 없이 다시 쓴 질문 한 줄만 출력해.\n\n"
)


def _connect() -> sqlite3.Connection:
    cache.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache.CACHE_PATH, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversation_turn (session TEXT, turn INTEGER, question TEXT, "
        "standalone TEXT, answer TEXT, created_at REAL, PRIMARY KEY (session, turn))"
    )
    return conn


def load_history(session: str, limit: int = _HISTORY_TURNS) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT turn, question, standalone, answer FROM conversation_turn WHERE session=? ORDER BY turn DESC LIMIT ?",
            (session, limit),
        ).fetchall()
    finally:
        conn.close()
    return [dict(zip(("turn", "question", "standalone", "answer"), r)) for r in reversed(rows)]


def _save_turn(session: str, question: str, standalone: str, answer: str) -> None:
    conn = _connect()
    try:
        (last,) = conn.execute("SELECT COALESCE(MAX(turn), 0) FROM conversation_turn WHERE session=?", (session,)).fetchone()
        conn.execute(
            "INSERT INTO conversation_turn VALUES (?, ?, ?, ?, ?, ?)",
            (session, last + 1, question, standalone, answer, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def rewrite_question(question: str, history: list[dict], usage: dict) -> str:
    """이전 대화가 없으면 LLM을 부르지 않고 그대로 돌려준다."""
    if not history:
        return question
    lines = []
    for h in history:
        answer = h["answer"][:_ANSWER_CHARS] + ("…" if len(h["answer"]) > _ANSWER_CHARS else "")
        lines.append(f"사용자: {h['standalone']}\n어시스턴트: {answer}")
    prompt = _REWRITE_PROMPT + "[이전 대화]\n" + "\n\n".join(lines) + f"\n\n[새 질문]\n{question}"
    resp = _get_client().chat.completions.create(model=PLANNER_MODEL, messages=[{"role": "user", "content": prompt}])
    record_usage(usage, resp, PLANNER_MODEL)
    return (resp.choices[0].message.content or question).strip()


def ask(session: str, question: str) -> dict:
    """대화를 이어서 질문한다. run_agent 결과에 "standalone_question"(다시 쓴 질문)을 더해 반환."""
    usage: dict = {}
    standalone = rewrite_question(question, load_history(session), usage)
    result = run_agent(standalone)
    for model, u in usage.items():  # 재작성 비용도 이번 턴 사용량에 합친다
        acc = result["usage"].setdefault(model, {"calls": 0, "prompt": 0, "completion": 0, "cached": 0})
        for k in acc:
            acc[k] += u[k]
    _save_turn(session, question, standalone, result["answer"])
    return {**result, "standalone_question": standalone}
