#!/bin/bash
# TAPT부터 앙상블까지 전 과정을 자동화하는 SOTA 워크플로우 스크립트
# transformers 라이브러리 업데이트로 인한 모델 저장 파일 형식 변경(safetensors)에 대응
# TAPT에서 AEDA 데이터셋 제외, FT에서 koelectra-small 모델 제외
# GPU 자원 교착 상태 해결을 위해 모든 학습을 순차 실행하고, 각 단계의 성공 여부를 자동으로 검증하도록 안정성 강화

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

# --- 변수 설정 ---
NIKL_DATASET="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
NIKL_REVISION="v1.2"
TAPT_EPOCHS=2

TAPT_MODEL_DIR="../tapt_models"
FT_MODEL_DIR="../ft_models"
LOG_DIR="./logs"

# --- 환경 설정 ---
mkdir -p $TAPT_MODEL_DIR $FT_MODEL_DIR $LOG_DIR

# --- 함수 정의 ---
# TAPT를 실행하고 결과를 검증하는 함수
run_tapt() {
    local base_model=$1
    local dataset=$2
    local revision=$3
    local epochs=$4
    local save_dir_name=$5
    local save_path="$TAPT_MODEL_DIR/$save_dir_name"
    local log_file="$LOG_DIR/tapt_$save_dir_name.log"

    echo "--- TAPT 시작: $base_model on $dataset (rev: $revision) ---"
    python tapt.py --base_model_name "$base_model" --dataset_name "$dataset" --dataset_revision "$revision" --epochs "$epochs" --output_model_path "$save_path" > "$log_file" 2>&1
    
    # [수정] TAPT 결과 검증: pytorch_model.bin 또는 model.safetensors 파일이 있는지 확인
    if [ -d "$save_path" ] && { [ -f "$save_path/pytorch_model.bin" ] || [ -f "$save_path/model.safetensors" ]; }; then
        echo "--- TAPT 성공: $save_dir_name ---"
    else
        echo "--- TAPT 실패: $save_dir_name. 모델 파일(pytorch_model.bin 또는 model.safetensors)이 생성되지 않았습니다. 로그 파일($log_file)을 확인하세요. ---"
        exit 1
    fi
}

# Fine-tuning을 실행하고 결과를 검증하는 함수
run_ft() {
    local base_model=$1
    local save_dir_name=$2
    local save_path="$FT_MODEL_DIR/$save_dir_name"
    local log_file="$LOG_DIR/ft_$save_dir_name.log"

    echo "--- Fine-tuning 시작: $base_model ---"
    python main.py --model_name "$base_model" --dataset_name "$NIKL_DATASET" --dataset_revision "$NIKL_REVISION" --output_dir "$save_path" --run_name "$save_dir_name" > "$log_file" 2>&1
    
    # FT 결과 검증
    if [ -d "$save_path" ] && [ -n "$(find "$save_path" -type d -name "checkpoint-*" | head -n 1)" ]; then
        echo "--- Fine-tuning 성공: $save_dir_name ---"
    else
        echo "--- Fine-tuning 실패: $save_dir_name. 로그 파일($log_file)을 확인하세요. ---"
        exit 1
    fi
}

# ===================================================================================
#                                  Phase 1: TAPT
# ===================================================================================
echo "Phase 1: Task-Adaptive Pre-training 시작"

# NIKL 데이터셋으로 TAPT 수행
run_tapt "klue/bert-base" "$NIKL_DATASET" "$NIKL_REVISION" "$TAPT_EPOCHS" "tapt-bert-nikl"
run_tapt "beomi/kcbert-base" "$NIKL_DATASET" "$NIKL_REVISION" "$TAPT_EPOCHS" "tapt-beomi-kcbert-nikl"

echo "Phase 1: TAPT 완료"

# ===================================================================================
#                                Phase 2: Fine-Tuning
# ===================================================================================
echo "Phase 2: Fine-tuning 시작"

# 기본 모델 4종 Fine-tuning
run_ft "klue/bert-base" "ft-bert-base"
run_ft "beomi/kcbert-base" "ft-beomi-kcbert-base"
run_ft "monologg/koelectra-base-v3-discriminator" "ft-koelectra-base"
run_ft "google/electra-base-discriminator" "ft-electra-base"

# TAPT 적용 모델 Fine-tuning
run_ft "$TAPT_MODEL_DIR/tapt-bert-nikl" "tapt-bert-nikl-ft"
run_ft "$TAPT_MODEL_DIR/tapt-beomi-kcbert-nikl" "tapt-beomi-kcbert-nikl-ft"

echo "Phase 2: Fine-tuning 완료"

# ===================================================================================
#                                 Phase 3: Ensemble
# ===================================================================================
echo "Phase 3: Ensemble 시작"

# 앙상블할 모델 경로 탐색 (오류 수정을 위해 `/results` 제거)
FT_BERT_PATH=$(find "$FT_MODEL_DIR/ft-bert-base" -type d -name "checkpoint-*" | head -n 1)
FT_BEOMI_KCBERT_PATH=$(find "$FT_MODEL_DIR/ft-beomi-kcbert-base" -type d -name "checkpoint-*" | head -n 1)
FT_KOELECTRA_BASE_PATH=$(find "$FT_MODEL_DIR/ft-koelectra-base" -type d -name "checkpoint-*" | head -n 1)
FT_ELECTRA_PATH=$(find "$FT_MODEL_DIR/ft-electra-base" -type d -name "checkpoint-*" | head -n 1)
TAPT_BERT_NIKL_FT_PATH=$(find "$FT_MODEL_DIR/tapt-bert-nikl-ft" -type d -name "checkpoint-*" | head -n 1)
TAPT_BEOMI_KCBERT_NIKL_FT_PATH=$(find "$FT_MODEL_DIR/tapt-beomi-kcbert-nikl-ft" -type d -name "checkpoint-*" | head -n 1)

# 전략 1: 기본 모델 4종 앙상블
echo "-> 전략 1: 기본 모델 4종 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_baseline_4models.csv" \
    --model_paths "$FT_BERT_PATH" "$FT_BEOMI_KCBERT_PATH" "$FT_ELECTRA_PATH" "$FT_KOELECTRA_BASE_PATH"

# 전략 2: TAPT (NIKL) 적용 모델 앙상블
echo "-> 전략 2: TAPT(NIKL) 적용 모델 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_nikl.csv" \
    --model_paths "$TAPT_BERT_NIKL_FT_PATH" "$TAPT_BEOMI_KCBERT_NIKL_FT_PATH" "$FT_KOELECTRA_BASE_PATH" "$FT_ELECTRA_PATH"

echo "Phase 3: Ensemble 완료"
echo "--- 모든 워크플로우가 성공적으로 완료되었습니다. ---"