#!/bin/bash
# ============================================================
# run_cA_unseen.sh
#
# 필터링 전 원본 unseen set으로 Candidate Answer Set 생성
#   - RealtimeQA (629문항) x 4 prompts
#   - Unseen Wiki (878문항) x 4 prompts
#
# 출력 폴더: candidate_answer_set_prompt{P}_rebuttal/
#   (기존 candidate_answer_set_prompt{P}/ 와 분리)
#
# 재개(resume):
#   이미 완성된 출력 파일은 모델 로딩 전에 건너뛰고, 중간에 끊긴 파일은
#   완료된 문항 다음부터 이어서 생성한다 (cA_mcq.py 가 문항 단위로 판단).
#   처음부터 다시 만들려면 아래 python 호출에 --overwrite 를 추가.
#
# Usage:
#   bash run_cA_unseen.sh <MODEL_SIZE> <GPU_ID>
#
# Examples:
#   bash run_cA_unseen.sh 7B 0
#   bash run_cA_unseen.sh 13B 1
#   bash run_cA_unseen.sh 32B 2,3
#
# 모든 모델 순차 실행:
#   for SIZE in 7B 13B 32B; do bash run_cA_unseen.sh $SIZE 0; done
#
# 실패 시 자동 재시도는 기본 꺼져 있음(MAX_RETRY=1). 과열로 죽으면 그대로 멈추고,
# 온도가 내려간 뒤 같은 명령을 다시 실행하면 끊긴 지점부터 이어서 생성한다.
# ============================================================

set -euo pipefail

MODEL_SIZE="${1:?Usage: bash run_cA_unseen.sh <MODEL_SIZE: 7B|13B|32B> <GPU_ID>}"
GPU_ID="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CA_SCRIPT="${SCRIPT_DIR}/candidate_answer_set/cA_mcq.py"

REALTIME_INPUT="/mnt/PK/dataset/realtimeqa_public/realtimeqa.jsonl"
WIKI_INPUT="/mnt/PK/dataset/current_events/unseen_wiki/wiki_event_only_clean.jsonl"
# 신규 unseen set (unseen_crawl/ 파이프라인 산출물, 2025-03-26~2026-07-29 크롤)
WIKI2026_INPUT="/mnt/PK/dataset/current_events/unseen_wiki_2026/unseen_wiki_2026.jsonl"

TAG="rebuttal"
# 기본값 1 = 재시도 없음. 실패하면 즉시 중단하고 GPU 를 쉬게 한다.
# (온도가 안전할 때 사람이 직접 다시 실행 → 문항 단위로 이어서 생성됨)
MAX_RETRY="${MAX_RETRY:-1}"

echo "============================================================"
echo " Candidate Answer Set Generation (Unfiltered Unseen)"
echo " Model: ${MODEL_SIZE}   GPU: ${GPU_ID}   Tag: ${TAG}"
echo " Resume: on (완료된 파일은 skip, 끊긴 파일은 이어서 생성)"
echo "============================================================"

# 실패 시 기본 동작은 즉시 중단(MAX_RETRY=1).
# 서버실 온도가 여유로울 때만 MAX_RETRY=3 bash run_cA_unseen.sh ... 처럼 켤 것.
run_step() {
    local attempt=1
    while true; do
        if CUDA_VISIBLE_DEVICES=${GPU_ID} python "${CA_SCRIPT}" "$@"; then
            return 0
        fi
        if [ "${attempt}" -ge "${MAX_RETRY}" ]; then
            echo "❌ ${MAX_RETRY}회 시도 후 실패: $*" >&2
            return 1
        fi
        attempt=$((attempt + 1))
        echo "⚠️  실패 — 60초 후 재시도 (${attempt}/${MAX_RETRY})" >&2
        sleep 60
    done
}

for PROMPT_ID in 4; do
    echo ""
    echo "------------------------------------------------------------"
    echo "[${MODEL_SIZE}] prompt_id=${PROMPT_ID} | RealtimeQA (629 questions)"
    echo "------------------------------------------------------------"
    run_step \
        --model_size "${MODEL_SIZE}" \
        --input_jsonl "${REALTIME_INPUT}" \
        --prompt_id "${PROMPT_ID}" \
        --realtimeQA \
        --tag "${TAG}"

    echo ""
    echo "------------------------------------------------------------"
    echo "[${MODEL_SIZE}] prompt_id=${PROMPT_ID} | Unseen Wiki (878 questions)"
    echo "------------------------------------------------------------"
    run_step \
        --model_size "${MODEL_SIZE}" \
        --input_jsonl "${WIKI_INPUT}" \
        --prompt_id "${PROMPT_ID}" \
        --tag "${TAG}"

    echo ""
    echo "------------------------------------------------------------"
    echo "[${MODEL_SIZE}] prompt_id=${PROMPT_ID} | Unseen Wiki 2026 (1520 questions)"
    echo "------------------------------------------------------------"
    run_step \
        --model_size "${MODEL_SIZE}" \
        --input_jsonl "${WIKI2026_INPUT}" \
        --prompt_id "${PROMPT_ID}" \
        --tag "${TAG}"
done

echo ""
echo "============================================================"
echo " Done: ${MODEL_SIZE} all prompts completed"
echo "============================================================"
