# stock-research-agent

멀티홉 에이전틱 RAG — 종목 리서치 어시스턴트

사용자의 복합 질문("A사 실적이 왜 안 좋았어?")을 LLM 플래너가 하위 질문으로 쪼개고,
DART 공시 API와 뉴스 검색이라는 서로 다른 도구를 상황에 맞게 호출해 종합 답변을
생성하는 에이전틱 RAG 프로젝트입니다.

전체 설계/계획: [Notion 계획서](https://app.notion.com/p/3e43a009bcdf8178861eecf2eb6ad11e)

## 진행 단계 (Phase)

- [x] Phase 1 — DART API 연동, 재무 데이터 수집 및 로컬 DB 구축
- [x] Phase 2 — 뉴스 API 연동 (실시간 검색 도구 + 평가용 고정 코퍼스)
- [x] Phase 3 — 단일 도구 RAG 베이스라인 (DART만 / News만)
- [ ] Phase 4 — Planner + Tool use 결합한 멀티홉 에이전트
- [ ] Phase 5 — 채점 파이프라인 구축 + ablation 실험

## 디렉토리 구조

```
src/
  data/     # DART / 뉴스 API 클라이언트, 수집 스크립트
  agent/    # Planner, Tool 정의, Aggregator
  eval/     # 채점 스크립트
tests/
notebooks/  # 탐색적 분석, 프로토타이핑
```

## 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env   # 이후 .env에 API 키 입력
```

## 필요한 API 키

`.env.example` 참고. 신청 방법은 프로젝트 노트/README 하단 참고.
