#!/bin/bash
# ============================================================
# run_all.sh
#
# unseen 데이터셋 신규 구축 파이프라인
#   크롤링 → MCQ 생성 → 주관식 답변 → judge → miss 추출
#   → 3모델 교집합 → 옵션 재배치 → publish
#
# Usage:
#   bash run_all.sh <STEP> [GPU_ID]
#
# Steps:
#   0_crawl       Portal:Current events 크롤링 (CPU)
#   1_gen         2-turn MCQ 생성 (GPU, Qwen2.5-32B)
#   2_filter      파싱 + grounding 필터 (CPU)
#   3_dedup       fuzzy 중복 제거 (CPU, sypk env)
#   4_answer      주관식 답변 생성 7B/13B/32B (GPU)
#   5_judge       정오 판정, 휴리스틱 (CPU)
#   6_split       hit/miss 분리 (CPU)
#   7_intersect   3모델 교집합 (CPU)
#   8_rebalance   정답 위치 재배치 (CPU)
#   9_publish     /mnt/PK/dataset 으로 배포 + 계약 검증 (CPU)
#   publish_pool  필터링 전 MCQ 풀을 배포 (다른 모델 실험의 출발점)
#   filter_with   기존 MCQ 풀에 대해 4~9 만 재실행 (다른 모델로 필터링)
#   smoke         20문항으로 1~9 전체 시험 실행
#   all           0~9 순차 실행
#
# Examples:
#   bash run_all.sh 0_crawl
#   bash run_all.sh smoke 0,1
#   bash run_all.sh all 0,1
#
# 스텝별로 필요한 conda env 의 python 을 절대경로로 직접 호출한다.
# (run_eval_unseen.sh 와 같은 방식)
# ============================================================

set -euo pipefail

STEP="${1:?Usage: bash run_all.sh <STEP> [GPU_ID]}"
GPU_ID="${2:-0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-${SCRIPT_DIR}/work}"

CONDA_ENVS="${CONDA_ENVS:-/home/jiyoung/miniconda3/envs}"
PY_PKEX="${CONDA_ENVS}/pkex/bin/python"     # 크롤링 (requests + bs4)
PY_OLMO2="${CONDA_ENVS}/olmo2/bin/python"   # vLLM 0.13 + sm_120 지원 torch
PY_SYPK="${CONDA_ENVS}/sypk/bin/python"     # rapidfuzz

GEN_MODEL="${GEN_MODEL:-/mnt/PK/models/Qwen/Qwen2.5-32B-Instruct}"
CRAWL_START="${CRAWL_START:-2025-03-26}"
CRAWL_END="${CRAWL_END:-2026-07-29}"

DEST_DIR="${DEST_DIR:-/mnt/PK/dataset/current_events/unseen_wiki_2026}"
STEM="${STEM:-unseen_wiki_2026}"

# 필터링에 쓰는 모델 집합.
#
# unseen 판정 기준은 "이 집합의 모든 모델이 주관식으로 틀린 문항"이다. 따라서
# 평가하려는 모델이 바뀌면 그 모델로 다시 필터링해야 한다 (OLMo-2 가 틀린 문항이
# 다른 모델에게도 unseen 이라는 보장은 없다).
#
# 다른 모델로 필터링하려면 ANSWER_MODELS 를 "라벨:경로[:tp]" 목록으로 넘긴다:
#
#   ANSWER_MODELS="qwen32b:/mnt/PK/models/Qwen/Qwen2.5-32B-Instruct:2" \
#   STEM=unseen_wiki_2026_qwen32b \
#   bash run_all.sh filter_with 0,1
#
ANSWER_MODELS="${ANSWER_MODELS:-\
7B:/mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct:1 \
13B:/mnt/PK/models/OLMo2/OLMo-2-1124-13B-Instruct:1 \
32B:/mnt/PK/models/OLMo2/OLMo-2-0325-32B-Instruct:2}"

# 교집합 조건: 몇 개 모델이 틀려야 unseen 으로 볼지. 0 = 전부.
REQUIRE="${REQUIRE:-0}"

model_labels() {
    local spec
    for spec in $ANSWER_MODELS; do echo "${spec%%:*}"; done
}
model_path_of() { local s="${1#*:}"; echo "${s%%:*}"; }
model_tp_of() {
    local tp="${1##*:}"
    # 필드가 2개뿐이면 tp 가 생략된 것 -> 1
    [ "$(echo "$1" | tr -cd ':' | wc -c)" -ge 2 ] && echo "$tp" || echo 1
}

log() { echo -e "\n==== $1 ===="; }

require_py() {
    local py="$1" label="$2"
    if [ ! -x "$py" ]; then
        echo "[ERROR] ${label}에 필요한 conda env python이 없습니다: ${py}" >&2
        exit 1
    fi
    echo "[env] ${label} -> ${py}"
}

# GPU 4~7 은 다른 작업이 점유 중이다 (2026-07-30 기준 84GB/99%).
# 실수로 넘기면 남의 작업을 OOM 시키므로 아예 거부한다.
guard_gpu() {
    case ",${GPU_ID}," in
        *,4,*|*,5,*|*,6,*|*,7,*)
            echo "[ERROR] GPU 4-7은 다른 작업이 사용 중입니다 (GPU_ID=${GPU_ID})" >&2
            echo "        0~3번만 사용하세요." >&2
            exit 1 ;;
    esac
}

# craft env 는 이 머신에서 동작하지 않는다:
# torch 2.2.1 에 sm_120(Blackwell) 커널이 없다.
if [ -n "${USE_CRAFT_ENV:-}" ]; then
    echo "[ERROR] craft env는 sm_120 비호환입니다 (torch 2.2.1). olmo2 env를 쓰세요." >&2
    exit 1
fi

# ============================================================
step_0_crawl() {
    log "Step 0: crawl Portal:Current events (${CRAWL_START} ~ ${CRAWL_END})"
    require_py "$PY_PKEX" "0_crawl"
    "$PY_PKEX" "${SCRIPT_DIR}/crawl_current_events.py" \
        --start "$CRAWL_START" --end "$CRAWL_END" \
        --out-dir "${WORK}/00_corpus" --cache-dir "${SCRIPT_DIR}/cache"
    "$PY_PKEX" "${SCRIPT_DIR}/verify_crawl.py" \
        --new "${WORK}/00_corpus/wiki_event_only.jsonl" \
        --meta "${WORK}/00_corpus/wiki_event_meta.jsonl"
}

step_1_gen() {
    log "Step 1: MCQ generation (GPU: ${GPU_ID})"
    guard_gpu
    require_py "$PY_OLMO2" "1_gen"
    "$PY_OLMO2" "${SCRIPT_DIR}/gen_mcq.py" \
        --corpus "${WORK}/00_corpus/wiki_event_only.jsonl" \
        --out_raw "${WORK}/10_gen/mcq_raw.jsonl" \
        --model "$GEN_MODEL" --gpu "$GPU_ID" --tp 2 ${GEN_EXTRA:-}
}

step_2_filter() {
    log "Step 2: parse + grounding filter"
    require_py "$PY_OLMO2" "2_filter"
    "$PY_OLMO2" "${SCRIPT_DIR}/filter_mcq.py" \
        --in "${WORK}/10_gen/mcq_raw.jsonl" \
        --out "${WORK}/20_filter/mcq_valid.jsonl" \
        --errors_csv "${WORK}/20_filter/mcq_errors.csv" \
        --stats_out "${WORK}/20_filter/filter_stats.json"
}

step_3_dedup() {
    log "Step 3: fuzzy dedup"
    require_py "$PY_SYPK" "3_dedup"
    "$PY_SYPK" "${SCRIPT_DIR}/dedup_mcq.py" \
        --in "${WORK}/20_filter/mcq_valid.jsonl" \
        --out "${WORK}/30_dedup/mcq_clean.jsonl" \
        --stats_out "${WORK}/30_dedup/dedup_stats.json"
}

step_4_answer() {
    log "Step 4: open-ended answering (GPU: ${GPU_ID})"
    guard_gpu
    require_py "$PY_OLMO2" "4_answer"
    local spec label path tp
    for spec in $ANSWER_MODELS; do
        label="${spec%%:*}"
        path="$(model_path_of "$spec")"
        tp="$(model_tp_of "$spec")"
        echo "[answer] ${label} (tp=${tp}) ${path}"
        "$PY_OLMO2" "${SCRIPT_DIR}/answer_openended.py" \
            --input "${WORK}/30_dedup/mcq_clean.jsonl" \
            --output "${WORK}/40_answers/cands_${label}.jsonl" \
            --model_label "$label" --model_path "$path" \
            --gpu "$GPU_ID" --tp "$tp"
    done
}

step_5_judge() {
    log "Step 5: judge (heuristic only)"
    require_py "$PY_OLMO2" "5_judge"
    mkdir -p "${WORK}/50_judged"
    local label
    for label in $(model_labels); do
        echo "[judge] ${label}"
        "$PY_OLMO2" "${SCRIPT_DIR}/judge_candidates.py" \
            --input "${WORK}/40_answers/cands_${label}.jsonl" \
            --output "${WORK}/50_judged/judged_${label}.jsonl" \
            --backend "${JUDGE_BACKEND:-none}"
    done
}

step_6_split() {
    log "Step 6: hit/miss split"
    require_py "$PY_OLMO2" "6_split"
    mkdir -p "${WORK}/60_split"
    local label
    for label in $(model_labels); do
        echo "[split] ${label}"
        "$PY_OLMO2" "${SCRIPT_DIR}/split_hits_misses.py" \
            --input "${WORK}/50_judged/judged_${label}.jsonl" \
            --hits_out "${WORK}/60_split/hits_${label}.jsonl" \
            --filtered_out "${WORK}/60_split/filtered_${label}.jsonl" \
            --misses_only_out "${WORK}/60_split/misses_${label}.jsonl"
    done
}

step_7_intersect() {
    log "Step 7: intersection over $(model_labels | tr '\n' ' ')"
    require_py "$PY_OLMO2" "7_intersect"
    local labels=() misses=() label
    for label in $(model_labels); do
        labels+=("$label")
        misses+=("${WORK}/60_split/misses_${label}.jsonl")
    done
    "$PY_OLMO2" "${SCRIPT_DIR}/intersect_misses.py" \
        --misses "${misses[@]}" \
        --labels "${labels[@]}" \
        --require "$REQUIRE" \
        --source "${WORK}/30_dedup/mcq_clean.jsonl" \
        --out "${WORK}/70_intersect/unseen_3way.jsonl" \
        --venn_out "${WORK}/70_intersect/venn.json"
}

# 이미 만들어둔 MCQ 풀(30_dedup/mcq_clean.jsonl)에 대해 필터링만 다시 돌린다.
# 다른 모델로 unseen 을 만들 때 쓴다 -- 크롤링/생성은 재사용한다.
step_filter_with() {
    log "Re-filtering the existing MCQ pool with: $(model_labels | tr '\n' ' ')"
    if [ ! -f "${WORK}/30_dedup/mcq_clean.jsonl" ]; then
        echo "[ERROR] MCQ 풀이 없습니다: ${WORK}/30_dedup/mcq_clean.jsonl" >&2
        echo "        먼저 0_crawl ~ 3_dedup 을 실행하세요." >&2
        exit 1
    fi
    step_4_answer
    step_5_judge
    step_6_split
    step_7_intersect
    step_8_rebalance
    step_9_publish
}

step_8_rebalance() {
    log "Step 8: rebalance answer positions"
    require_py "$PY_OLMO2" "8_rebalance"
    "$PY_OLMO2" "${SCRIPT_DIR}/rebalance_options.py" \
        --in "${WORK}/70_intersect/unseen_3way.jsonl" \
        --out "${WORK}/80_rebalance/${STEM}.jsonl" \
        --stats_out "${WORK}/80_rebalance/rebalance_stats.json"
}

step_9_publish() {
    log "Step 9: verify + publish"
    require_py "$PY_OLMO2" "9_publish"
    "$PY_OLMO2" "${SCRIPT_DIR}/verify_dataset.py" \
        --in "${WORK}/80_rebalance/${STEM}.jsonl"
    "$PY_OLMO2" "${SCRIPT_DIR}/publish_dataset.py" \
        --in "${WORK}/80_rebalance/${STEM}.jsonl" \
        --dest_dir "$DEST_DIR" --stem "$STEM"
}

# 필터링 전 MCQ 풀을 그대로 배포한다.
#
# 다른 모델로 unseen 을 만들려면 이 풀에서 시작해야 한다. 옵션 순서만 균등화해서
# 배포하는데, 주관식 필터링은 보기를 보여주지 않고 정답도 옵션 "텍스트"로 잡으므로
# 옵션 순서에 전혀 영향받지 않는다. 반면 이 풀로 MCQ 정확도를 재려면 균등화가
# 필수다 (원본은 B 가 64%).
step_publish_pool() {
    log "Publish the unfiltered MCQ pool"
    require_py "$PY_OLMO2" "publish_pool"
    local stem="${POOL_STEM:-unseen_wiki_2026_pool}"
    "$PY_OLMO2" "${SCRIPT_DIR}/rebalance_options.py" \
        --in "${WORK}/30_dedup/mcq_clean.jsonl" \
        --out "${WORK}/30_dedup/${stem}.jsonl" \
        --stats_out "${WORK}/30_dedup/pool_rebalance_stats.json"
    "$PY_OLMO2" "${SCRIPT_DIR}/verify_dataset.py" \
        --in "${WORK}/30_dedup/${stem}.jsonl"
    "$PY_OLMO2" "${SCRIPT_DIR}/publish_dataset.py" \
        --in "${WORK}/30_dedup/${stem}.jsonl" \
        --dest_dir "$DEST_DIR" --stem "$stem"
}

# 20문항으로 1~9 전 구간을 돌려본다. 실물 실행 전 반드시 통과시킬 것.
step_smoke() {
    log "SMOKE: 20 bullets end-to-end"
    WORK="${SCRIPT_DIR}/work/smoke" GEN_EXTRA="--limit 20" \
    DEST_DIR="${SCRIPT_DIR}/work/smoke/publish" \
        bash "${BASH_SOURCE[0]}" _smoke_inner "$GPU_ID"
}

step__smoke_inner() {
    mkdir -p "${WORK}/00_corpus"
    head -20 "${SCRIPT_DIR}/work/00_corpus/wiki_event_only.jsonl" \
        > "${WORK}/00_corpus/wiki_event_only.jsonl"
    step_1_gen
    step_2_filter
    step_3_dedup
    step_4_answer
    step_5_judge
    step_6_split
    step_7_intersect
    step_8_rebalance
    # 표본이 작아 정답 문자 분포는 25%에 못 맞으므로 허용오차를 넓힌다.
    "$PY_OLMO2" "${SCRIPT_DIR}/verify_dataset.py" \
        --in "${WORK}/80_rebalance/${STEM}.jsonl" --letter_tol 0.5
    log "SMOKE done -- inspect ${WORK} before running 'all'"
}

case "$STEP" in
    0_crawl)      step_0_crawl ;;
    1_gen)        step_1_gen ;;
    2_filter)     step_2_filter ;;
    3_dedup)      step_3_dedup ;;
    4_answer)     step_4_answer ;;
    5_judge)      step_5_judge ;;
    6_split)      step_6_split ;;
    7_intersect)  step_7_intersect ;;
    8_rebalance)  step_8_rebalance ;;
    9_publish)    step_9_publish ;;
    filter_with)  step_filter_with ;;
    publish_pool) step_publish_pool ;;
    smoke)        step_smoke ;;
    _smoke_inner) step__smoke_inner ;;
    all)
        step_0_crawl
        step_1_gen
        step_2_filter
        step_3_dedup
        step_4_answer
        step_5_judge
        step_6_split
        step_7_intersect
        step_8_rebalance
        step_9_publish
        ;;
    *)
        echo "Unknown step: $STEP" >&2
        echo "Available: 0_crawl 1_gen 2_filter 3_dedup 4_answer 5_judge 6_split 7_intersect 8_rebalance 9_publish smoke all" >&2
        exit 1 ;;
esac

echo ""
log "Done: step=${STEP}"
