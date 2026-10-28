# unseen_crawl

Builds a fresh **unseen** MCQ evaluation set from Wikipedia's
*Portal:Current events*, keeping only the questions that **OLMo-2 7B, 13B and
32B all fail** in open-ended (no-options) form.

Replaces the previous `unseen_wiki` set, whose crawler lived on another server
(`aix23606`) and was lost.

```
crawl → MCQ 생성 → grounding 필터 → dedup
      → 주관식 답변(7B/13B/32B) → judge → miss 추출
      → 3모델 교집합 → 정답위치 재배치 → publish
```

## Quick start

```bash
bash run_all.sh 0_crawl            # ~8분, CPU
bash run_all.sh smoke 0,1          # 20문항 전 구간 시험 — 먼저 통과시킬 것
bash run_all.sh all 0,1            # 전체
```

`GPU_ID`는 **0~3만** 허용된다. 4~7번은 다른 작업이 점유 중이라 스크립트가 거부한다.

## 다른 모델로 unseen 을 만들 때

unseen 판정은 **"필터링에 쓴 모델들이 모두 주관식으로 틀린 문항"** 이다. 따라서 평가 대상
모델이 바뀌면 **그 모델로 다시 필터링해야** 한다 — OLMo-2 가 틀린 문항이 다른 모델에게도
unseen 이라는 보장은 없다.

크롤링·MCQ 생성은 모델과 무관하므로 재사용한다. 출발점은 **필터링 전 MCQ 풀**:

```
/mnt/PK/dataset/current_events/unseen_wiki_2026/unseen_wiki_2026_pool.jsonl   # 1,742문항
/mnt/PK/dataset/current_events/unseen_wiki_2026/unseen_wiki_2026.jsonl        # 1,520문항 (OLMo-2 3개 모델 필터)
```

`filter_with` 로 4~9 단계만 다시 돌린다:

```bash
ANSWER_MODELS="qwen32b:/mnt/PK/models/Qwen/Qwen2.5-32B-Instruct:2" \
STEM=unseen_wiki_2026_qwen32b \
bash run_all.sh filter_with 0,1
```

`ANSWER_MODELS` 는 `라벨:모델경로[:tp]` 를 공백으로 구분해 나열한다. 여러 모델을 넘기면
기본적으로 **전부 틀린 문항**만 남고, `REQUIRE=2` 처럼 지정하면 "2개 이상이 틀린 문항"이 된다.

**옵션 순서는 신경 쓸 필요 없다.** 주관식 필터링은 보기를 보여주지 않고 정답도 옵션
*텍스트*로 잡으므로 순서에 영향받지 않는다. 그래서 배포된 풀은 이미 정답 위치를 균등화해둔
상태이고(원본은 B 가 64%), 그대로 MCQ 정확도 측정에도 쓸 수 있다.

## Steps

| STEP | 산출물 | env |
|---|---|---|
| `0_crawl` | `work/00_corpus/wiki_event_only.jsonl` | pkex |
| `1_gen` | `work/10_gen/mcq_raw.jsonl` | olmo2 (GPU) |
| `2_filter` | `work/20_filter/mcq_valid.jsonl` | olmo2 |
| `3_dedup` | `work/30_dedup/mcq_clean.jsonl` ← **모델 무관 MCQ 풀** | **sypk** (rapidfuzz) |
| `publish_pool` | `unseen_wiki_2026_pool.jsonl` (풀 배포) | olmo2 |
| `4_answer` | `work/40_answers/cands_{7B,13B,32B}.jsonl` | olmo2 (GPU) |
| `5_judge` | `work/50_judged/judged_*.jsonl` | olmo2 |
| `6_split` | `work/60_split/{hits,filtered,misses}_*.jsonl` | olmo2 |
| `7_intersect` | `work/70_intersect/unseen_3way.jsonl` | olmo2 |
| `8_rebalance` | `work/80_rebalance/unseen_wiki_2026.jsonl` | olmo2 |
| `9_publish` | `/mnt/PK/dataset/current_events/unseen_wiki_2026/` | olmo2 |
| `filter_with` | 4~9 만 재실행 (다른 모델용) | mixed |

## 이 코드가 CRAFT를 그대로 쓰지 않는 이유

### 1. EVENT_ONLY 프롬프트가 dead code였다

`CRAFT_new/code/utils/ts_creation.py`에서 `class MetaInstructions`는 249~423행인데
`QA_MC_INSTRUCTION_EVENT_ONLY`(425행)와 `QA_MC_REVIEW_INSTRUCTION_EVENT_ONLY`(448행)는
**클래스 밖 module scope**에 있다. 그래서

```python
getattr(MetaInstructions, "QA_MC_INSTRUCTION_EVENT_ONLY",
        MetaInstructions.QA_MC_INSTRUCTION)   # create_task_samples_diff.py:169
```

이 **항상 fallback**해서, 일회성 사건이 아니라 정의·역할·소속·위치 같은 배경지식을
물으라고 지시하는 일반 프롬프트가 쓰였다. 기존 unseen_wiki가 그렇게 만들어졌다.
(확인: `python -c "import ast; ..."` — 아래 검증 절 참고)

여기서는 `prompts.py`의 모듈 상수를 이름으로 직접 참조하므로 오타가 나면 `ImportError`가
나지 silent fallback이 될 수 없고, `gen_mcq.py`가 시작할 때 프롬프트 첫 줄을 assert한다.

### 2. `craft` conda env가 이 머신에서 못 돈다

`craft`의 torch 2.2.1+cu121에는 sm_120(RTX PRO 6000 Blackwell) 커널이 없다.
`olmo2` env(torch 2.9.0+cu128, vLLM 0.13.0)로 이식했다. `run_all.sh`는 `USE_CRAFT_ENV`가
설정돼 있으면 거부한다.

### 3. 크롤러 정규화 재현

기존 코퍼스는 `action=parse&prop=text`(렌더링 HTML) → `div.current-events-content`의
**leaf `<li>`** → `li.get_text(separator=" ")`로 만들어졌다. `separator=" "` 때문에
`"led by AFRICOM ."`, `"( The Guardian )"` 같은 공백이 생기고 전체 줄의 ~81%가 이에 해당한다.
`verify_crawl.py`가 이 비율을 canary로 검사한다 — 0으로 붕괴하면 정규화가 바뀐 것이다.

## 검증

```bash
# 크롤 코퍼스 형태 검사 (기존 코퍼스와 분포/겹침 비교)
python verify_crawl.py --new work/00_corpus/wiki_event_only.jsonl \
    --meta work/00_corpus/wiki_event_meta.jsonl

# CRAFT 프롬프트 버그를 문서화하는 AST 검사
python - <<'PY'
import ast
t = ast.parse(open('/home/jiyoung/gayoung/CRAFT_new/code/utils/ts_creation.py').read())
cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == 'MetaInstructions')
members = [x.targets[0].id for x in cls.body if isinstance(x, ast.Assign)]
assert 'QA_MC_INSTRUCTION_EVENT_ONLY' not in members   # the bug
print(f'MetaInstructions spans {cls.lineno}-{cls.end_lineno}, EVENT_ONLY is module-level')
PY

# dedup 스코어러 자기검사
python dedup_mcq.py --selftest --in work/20_filter/mcq_valid.jsonl

# 최종 데이터셋이 cA_mcq.py 계약을 만족하는지
python verify_dataset.py --in work/80_rebalance/unseen_wiki_2026.jsonl
```

## lumos 연결

`publish_dataset.py`가 `/mnt/PK/dataset/...` 아래에 써야 하는 이유는
`cA_mcq.py:107-116`이 `--input_jsonl`이 `/mnt/PK/MCQ` 또는 `/mnt/PK/dataset`으로
시작하지 않으면 `ValueError`를 내기 때문이다. 경로에 `final`이라는 디렉토리명을 쓰면
`split_for_probing.py:51-55`가 `/final/AT_rebuttal_new/`로 재작성하므로 금지된다.

등록 후:

```bash
bash ../run_cA_unseen.sh 7B 0
bash ../run_eval_unseen.sh 1_split
```

## 이식한 원본

| 여기 | 원본 |
|---|---|
| `prompts.py` | `CRAFT_new/code/utils/ts_creation.py:425-479` |
| `mcq_common.py` | 같은 파일 `:64-153`, `:543-584`; `create_task_samples_diff.py:66-104`, `:363-370` |
| `gen_mcq.py` | `create_task_samples_diff.py:148-352` (event 분기만) |
| `answer_openended.py` | `/mnt/PK/dataset/current_events/unseen_wiki/CA_current_wiki.py` |
| `judge_candidates.py` | `/mnt/PK/dataset/realtimeqa_public/2.judge_realtime.py` (vendored, `--backend` 기본값만 변경) |
| `split_hits_misses.py` | `/mnt/PK/dataset/realtimeqa_public/3.extract_realtime.py` (vendored, 무수정) |
| `intersect_misses.py` | `최종추출.py`의 3모델 확장판 (신규) |
| `rebalance_options.py` | 신규 — 원본 스크립트가 유실됨 |
