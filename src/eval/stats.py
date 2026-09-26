"""평가 결과의 불확실성 계산 (API 호출 없음).

"5/6 대 4/6"은 문항 하나만 달라도 뒤집히는 차이라 그대로는 우열을 말할 수 없다. 문항을 다시 뽑는 부트스트랩으로
"두 방식 통과율 차이"의 95% 신뢰구간을 구해, 구간이 0을 포함하면 "차이가 있다고 말하기 어렵다"고 보고한다.
"""

import random


def per_question_rate(rows: list[dict], approach: str, field: str = "judge_pass") -> dict[str, float]:
    """문항별 통과율 (반복 실행이 있으면 반복 평균). 판정이 없는(None) 기록은 뺀다."""
    by_q: dict[str, list[bool]] = {}
    for r in rows:
        if r["approach"] == approach and r.get(field) is not None:
            by_q.setdefault(r["question_id"], []).append(bool(r[field]))
    return {q: sum(v) / len(v) for q, v in by_q.items()}


def paired_bootstrap_ci(
    diffs: list[float], n_boot: int = 5000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float] | None:
    """문항별 차이(같은 문항에서 A 통과율 - B 통과율)의 평균과 신뢰구간. 문항을 복원 추출로 다시 뽑는다."""
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((1 - level) / 2 * n_boot)]
    hi = means[int((1 + level) / 2 * n_boot) - 1]
    return sum(diffs) / n, lo, hi


def flip_count(rows: list[dict], approach: str, field: str = "judge_pass") -> tuple[int, int]:
    """반복 실행에서 판정이 한 번이라도 바뀐 문항 수 / 반복이 2번 이상 있는 문항 수."""
    by_q: dict[str, set[bool]] = {}
    counts: dict[str, int] = {}
    for r in rows:
        if r["approach"] == approach and r.get(field) is not None:
            by_q.setdefault(r["question_id"], set()).add(bool(r[field]))
            counts[r["question_id"]] = counts.get(r["question_id"], 0) + 1
    repeated = [q for q, c in counts.items() if c >= 2]
    return sum(len(by_q[q]) > 1 for q in repeated), len(repeated)
