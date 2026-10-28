#!/bin/bash
# ============================================================
# run_eval_unseen.sh
#
# 필터링 전 원본 unseen set 평가 파이프라인
# cA 생성(run_cA_unseen.sh) 완료 후 실행
#
# 폴더 규칙:
#   candidate_answer_set_prompt{P}_rebuttal/
#   internal_prompt{P}_rebuttal/
#   external_prompt{P}_rebuttal/
#
# Usage:
#   bash run_eval_unseen.sh <STEP> <GPU_ID>
#
# Steps:
#   1_split       Probing/Labeling 분할 (CPU)
#   2_normP       Normalized Probability 계산 (GPU)
#   3_metrics     Acc/NormP/Consistency/Correlation (CPU)
#   4_sv          Self-Verify (GPU+CPU)
#   5_saplma      SAPLMA train → infer → score (GPU)
#   6_mars        MARS score → summary (GPU)
#   7_linear      Linear Probing train → infer → score (GPU)
#   all           전체 순차 실행
#
# Examples:
#   bash run_eval_unseen.sh 1_split            # CPU 작업
#   bash run_eval_unseen.sh 2_normP 0          # GPU 0
#   bash run_eval_unseen.sh 5_saplma 1         # GPU 1
#   bash run_eval_unseen.sh all 0              # 전체 순차 실행
#
# Conda 환경:
#   스텝별로 필요한 env의 python을 절대경로로 직접 호출한다.
#   따라서 실행 전에 conda activate를 할 필요가 없고,
#   어떤 env에서 실행하든 각 스텝은 항상 지정된 env에서 돌아간다.
# ============================================================

set -euo pipefail

STEP="${1:?Usage: bash run_eval_unseen.sh <STEP> [GPU_ID]}"
GPU_ID="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IO="/mnt/PK/Inside-out"
TAG="_rebuttal"

SIZES=(7B 13B 32B) # 7B 13B 32B
PROMPTS=(4) # 1 2 3 4

# SAPLMA(5_saplma) 전용: prompt1에 대해서만 수행한다.
# PROMPTS를 어떻게 바꾸든 SAPLMA는 항상 이 값만 사용한다.
SAPLMA_PROMPTS=(1)

declare -A MODEL_PATHS
MODEL_PATHS[7B]="/mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct"
MODEL_PATHS[13B]="/mnt/PK/models/OLMo2/OLMo-2-1124-13B-Instruct"
MODEL_PATHS[32B]="/mnt/PK/models/OLMo2/OLMo-2-0325-32B-Instruct"

# 데이터셋: (서브디렉토리 파일명stem)
DATASETS=(
    "realtimeqa_public realtimeqa"
    "current_events/unseen_wiki_2026 unseen_wiki_2026"
) # "current_events/unseen_wiki wiki_event_only_clean"

log() { echo -e "\n==== $1 ===="; }

# ============================================================
# Conda 환경 (스텝별 고정)
#
#   pkex  : 1_split / 2_normP / 3_metrics / 4_sv
#   olmo2 : 5_saplma
#   mars   : 6_mars / 7_linear      (TruthTorchLM 설치된 env)
#
# conda activate 대신 env의 python 바이너리를 절대경로로 호출한다.
# activate는 현재 셸에만 적용되어 스텝별로 갈아끼울 수 없기 때문.
# ============================================================
CONDA_ENVS="${CONDA_ENVS:-/home/jiyoung/miniconda3/envs}"

PY_PKEX="${CONDA_ENVS}/pkex/bin/python"
PY_OLMO2="${CONDA_ENVS}/olmo2/bin/python"
PY_MARS="${CONDA_ENVS}/mars/bin/python"

# 스텝 시작 시점에 env 존재를 먼저 확인한다.
# 없으면 GPU에 모델을 올린 뒤 죽는 대신 즉시 실패시킨다.
require_py() {
    local py="$1"
    local label="$2"
    if [ ! -x "$py" ]; then
        echo "[ERROR] ${label}에 필요한 conda env python이 없습니다: ${py}" >&2
        echo "        (conda env list 로 확인하세요)" >&2
        exit 1
    fi
    echo "[env] ${label} -> ${py}"
}

# ============================================================
# Step 1: Probing/Labeling 분할 (CPU)
# ============================================================
step_1_split() {
    log "Step 1: split_for_probing"
    require_py "$PY_PKEX" "1_split"
    cd "${SCRIPT_DIR}/candidate_answer_set"

    for SIZE in "${SIZES[@]}"; do
      for P in "${PROMPTS[@]}"; do
        for PAIR in "${DATASETS[@]}"; do
          SUBDIR=$(echo "$PAIR" | cut -d' ' -f1)
          STEM=$(echo "$PAIR" | cut -d' ' -f2)
          INPUT="${IO}/${SIZE}/candidate_answer_set_prompt${P}${TAG}/${SUBDIR}/${STEM}.jsonl"

          if [ ! -f "$INPUT" ]; then
              echo "[SKIP] not found: $INPUT"
              continue
          fi

          echo "[split] ${SIZE} prompt${P} ${STEM}"
          "$PY_PKEX" split_for_probing.py \
              --input_jsonl "$INPUT" \
              --train_ratio 0.25
        done
      done
    done
}

# ============================================================
# Step 2: Normalized Probability (GPU)
# ============================================================
step_2_normP() {
    log "Step 2: step1_normP (GPU: ${GPU_ID})"
    require_py "$PY_PKEX" "2_normP"
    cd "${SCRIPT_DIR}/scoring/others"

    for SIZE in "${SIZES[@]}"; do
      for P in "${PROMPTS[@]}"; do
        for PAIR in "${DATASETS[@]}"; do
          SUBDIR=$(echo "$PAIR" | cut -d' ' -f1)
          STEM=$(echo "$PAIR" | cut -d' ' -f2)
          INPUT="${IO}/${SIZE}/internal_prompt${P}${TAG}/${SUBDIR}/${STEM}_labeling.jsonl"

          if [ ! -f "$INPUT" ]; then
              echo "[SKIP] not found: $INPUT"
              continue
          fi

          echo "[normP] ${SIZE} prompt${P} ${STEM}"
          CUDA_VISIBLE_DEVICES=${GPU_ID} "$PY_PKEX" step1_normP.py \
              --model_size "$SIZE" \
              --prompt_id "$P" \
              --input_jsonl "$INPUT"
        done
      done
    done
}

# ============================================================
# Step 3: Acc / NormP avg / Consistency / Correlation (CPU)
# ============================================================
step_3_metrics() {
    log "Step 3: step2_metrics_runner"
    require_py "$PY_PKEX" "3_metrics"
    cd "${SCRIPT_DIR}/scoring/others"

    for SIZE in "${SIZES[@]}"; do
      for P in "${PROMPTS[@]}"; do
        for PAIR in "${DATASETS[@]}"; do
          SUBDIR=$(echo "$PAIR" | cut -d' ' -f1)
          STEM=$(echo "$PAIR" | cut -d' ' -f2)
          INPUT="${IO}/${SIZE}/external_prompt${P}${TAG}/${SUBDIR}/${STEM}_labeling.jsonl"

          if [ ! -f "$INPUT" ]; then
              echo "[SKIP] not found: $INPUT"
              continue
          fi

          echo "[metrics] ${SIZE} prompt${P} ${STEM}"
          "$PY_PKEX" step2_metrics_runner.py --input_jsonl "$INPUT"
        done
      done
    done
}

# ============================================================
# Step 4: Self-Verify (GPU + CPU)
# ============================================================
step_4_sv() {
    log "Step 4: Self-Verify (GPU: ${GPU_ID})"
    require_py "$PY_PKEX" "4_sv"
    cd "${SCRIPT_DIR}/scoring/self-verify"

    for SIZE in "${SIZES[@]}"; do
      for P in "${PROMPTS[@]}"; do
        for PAIR in "${DATASETS[@]}"; do
          SUBDIR=$(echo "$PAIR" | cut -d' ' -f1)
          STEM=$(echo "$PAIR" | cut -d' ' -f2)
          INPUT="${IO}/${SIZE}/external_prompt${P}${TAG}/${SUBDIR}/${STEM}_labeling.jsonl"

          if [ ! -f "$INPUT" ]; then
              echo "[SKIP] not found: $INPUT"
              continue
          fi

          echo "[SV step1] ${SIZE} prompt${P} ${STEM}"
          CUDA_VISIBLE_DEVICES=${GPU_ID} "$PY_PKEX" step1_cA_for_SV.py \
              --model_size "$SIZE" \
              --prompt_id "$P" \
              --input_jsonl "$INPUT"

          JUDGE="${INPUT%.jsonl}_judge.jsonl"
          echo "[SV step2] ${SIZE} prompt${P} ${STEM}"
          "$PY_PKEX" step2_SV_confusion_matrix.py --input_jsonl "$JUDGE"
        done
      done
    done
}

# ============================================================
# Step 5: SAPLMA (GPU)
# ============================================================
step_5_saplma() {
    log "Step 5: SAPLMA (GPU: ${GPU_ID})"
    require_py "$PY_OLMO2" "5_saplma"
    cd "${SCRIPT_DIR}/scoring/SAPLMA"

    # SAPLMA는 prompt1에 대해서만 수행한다.
    # 상단 PROMPTS 값과 무관하게 항상 1로 고정한다.
    echo "[note] SAPLMA는 prompt1만 계산합니다 (PROMPTS=${PROMPTS[*]} 무시)"

    for SIZE in "${SIZES[@]}"; do
      MODEL="${MODEL_PATHS[$SIZE]}"
      for P in "${SAPLMA_PROMPTS[@]}"; do
        for PAIR in "${DATASETS[@]}"; do
          SUBDIR=$(echo "$PAIR" | cut -d' ' -f1)
          STEM=$(echo "$PAIR" | cut -d' ' -f2)
          BASE="${IO}/${SIZE}/internal_prompt${P}${TAG}/${SUBDIR}"
          PROBE="${BASE}/${STEM}_probing.jsonl"
          LABEL="${BASE}/${STEM}_labeling.jsonl"
          OUTDIR="${BASE}/saplma"

          if [ ! -f "$PROBE" ]; then
              echo "[SKIP] not found: $PROBE"
              continue
          fi

          echo "[SAPLMA train] ${SIZE} prompt${P} ${STEM}"
          CUDA_VISIBLE_DEVICES=${GPU_ID} "$PY_OLMO2" saplma_train.py \
              --data_jsonl "$PROBE" \
              --model_path "$MODEL" \
              --out_dir "$OUTDIR"

          echo "[SAPLMA infer] ${SIZE} prompt${P} ${STEM}"
          CUDA_VISIBLE_DEVICES=${GPU_ID} "$PY_OLMO2" saplma_infer.py \
              --data_jsonl "$LABEL" \
              --model_path "$MODEL" \
              --probe_dir "$OUTDIR" \
              --out_jsonl "${OUTDIR}/scored.jsonl"

          echo "[SAPLMA score] ${SIZE} prompt${P} ${STEM}"
          "$PY_OLMO2" saplma_score.py --input_jsonl "${OUTDIR}/scored.jsonl"
        done
      done
    done
}

# ============================================================
# Step 6: MARS (GPU)
# ============================================================
step_6_mars() {
    log "Step 6: MARS (GPU: ${GPU_ID})"
    require_py "$PY_MARS" "6_mars"
    cd "${SCRIPT_DIR}/scoring/MARS"

    for SIZE in "${SIZES[@]}"; do
      MODEL="${MODEL_PATHS[$SIZE]}"
      for P in "${PROMPTS[@]}"; do
        for PAIR in "${DATASETS[@]}"; do
          SUBDIR=$(echo "$PAIR" | cut -d' ' -f1)
          STEM=$(echo "$PAIR" | cut -d' ' -f2)
          INPUT="${IO}/${SIZE}/external_prompt${P}${TAG}/${SUBDIR}/${STEM}_labeling.jsonl"
          OUTPUT="${IO}/${SIZE}/external_prompt${P}${TAG}/${SUBDIR}/${STEM}_labeling_mars.jsonl"

          if [ ! -f "$INPUT" ]; then
              echo "[SKIP] not found: $INPUT"
              continue
          fi

          echo "[MARS score] ${SIZE} prompt${P} ${STEM}"
          CUDA_VISIBLE_DEVICES=${GPU_ID} \
          MODEL_NAME="$MODEL" \
          INPUT_FILE="$INPUT" \
          OUTPUT_FILE="$OUTPUT" \
          "$PY_MARS" mars_score.py

          echo "[MARS avg] ${SIZE} prompt${P} ${STEM}"
          "$PY_MARS" avg_mars.py "$OUTPUT"
        done
      done
    done
}

# ============================================================
# Step 7: Linear Probing (GPU)
# ============================================================
step_7_linear() {
    log "Step 7: Linear Probing (GPU: ${GPU_ID})"
    require_py "$PY_MARS" "7_linear"
    cd "${SCRIPT_DIR}/scoring/linear_probing"

    for SIZE in "${SIZES[@]}"; do
      MODEL="${MODEL_PATHS[$SIZE]}"
      for P in "${PROMPTS[@]}"; do
        for PAIR in "${DATASETS[@]}"; do
          SUBDIR=$(echo "$PAIR" | cut -d' ' -f1)
          STEM=$(echo "$PAIR" | cut -d' ' -f2)
          BASE="${IO}/${SIZE}/internal_prompt${P}${TAG}/${SUBDIR}"
          PROBE="${BASE}/${STEM}_probing.jsonl"
          LABEL="${BASE}/${STEM}_labeling.jsonl"
          OUTDIR="${BASE}/linear_probe"

          if [ ! -f "$PROBE" ]; then
              echo "[SKIP] not found: $PROBE"
              continue
          fi

          echo "[linear train] ${SIZE} prompt${P} ${STEM}"
          CUDA_VISIBLE_DEVICES=${GPU_ID} "$PY_MARS" mcq_factual_train.py \
              --data_jsonl "$PROBE" \
              --model_path "$MODEL" \
              --out_dir "$OUTDIR"

          echo "[linear infer] ${SIZE} prompt${P} ${STEM}"
          CUDA_VISIBLE_DEVICES=${GPU_ID} "$PY_MARS" mcq_factual_infer.py \
              --data_jsonl "$LABEL" \
              --model_path "$MODEL" \
              --probe_dir "$OUTDIR" \
              --out_jsonl "${OUTDIR}/scored.jsonl"

          echo "[linear avg] ${SIZE} prompt${P} ${STEM}"
          INPUT_JSONL="${OUTDIR}/scored.jsonl" \
          OUTPUT_JSONL="${OUTDIR}/output_accuracy_probe.jsonl" \
          "$PY_MARS" avg_linear.py
        done
      done
    done
}

# ============================================================
# Dispatcher
# ============================================================
case "$STEP" in
    1_split)   step_1_split ;;
    2_normP)   step_2_normP ;;
    3_metrics) step_3_metrics ;;
    4_sv)      step_4_sv ;;
    5_saplma)  step_5_saplma ;;
    6_mars)    step_6_mars ;;
    7_linear)  step_7_linear ;;
    all)
        step_1_split
        step_2_normP
        step_3_metrics
        step_4_sv
        # step_5_saplma
        step_6_mars
        step_7_linear
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Available: 1_split 2_normP 3_metrics 4_sv 5_saplma 6_mars 7_linear all"
        exit 1
        ;;
esac

echo ""
log "Done: step=${STEP}"
