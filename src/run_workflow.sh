#!/bin/bash
# TAPT부터 앙상블까지 전 과정을 자동화하는 SOTA 워크플로우 스크립트
# [v5] GPU 자원 교착 상태 해결을 위해 모든 학습을 순차 실행하고, 각 단계의 성공 여부를 자동으로 검증하도록 안정성 강화

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

# --- 변수 설정 ---
NIKL_DATASET="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
NIKL_REVISION="v1.2"
AEDA_DATASET="ensemble-2/AEDA-dataset"
AEDA_REVISION="v1.1"
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

    # TAPT 결과물(모델 파일)이 정상적으로 생성되었는지 검증
    if [ ! -f "$save_path/pytorch_model.bin" ] && [ ! -f "$save_path/model.safetensors" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: TAPT 실패! '$save_path'에 모델 파일이 생성되지 않았습니다."
        echo "자세한 원인은 로그 파일을 확인하세요: $log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    echo "--- TAPT 완료: $save_dir_name ---"
}

# Fine-tuning을 실행하고 결과를 검증하는 함수
run_finetune() {
    local model_name=$1
    local run_name=$2
    local lr=$3
    local save_dir_name=$4
    local save_path="$FT_MODEL_DIR/$save_dir_name"
    local log_file="$LOG_DIR/$run_name.log"

    echo "--- Fine-tuning 시작: $model_name ---"
    python main.py --model_name "$model_name" --run_name "$run_name" --lr "$lr" --save_path "$save_path" > "$log_file" 2>&1
    
    # Fine-tuning 결과물(체크포인트 폴더)이 정상적으로 생성되었는지 검증
    if [ ! -d "$save_path/results" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: Fine-tuning 실패! '$save_path/results' 디렉토리가 생성되지 않았습니다."
        echo "자세한 원인은 로그 파일을 확인하세요: $log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    echo "--- Fine-tuning 완료: $model_name ---"
}

# ======================================================
# PHASE 1: TAPT 시작 (순차 실행 및 결과 검증)
# ======================================================
echo "PHASE 1: Task-Adaptive Pre-training (TAPT) 시작"

run_tapt "klue/bert-base" $NIKL_DATASET $NIKL_REVISION $TAPT_EPOCHS "bert_on_nikl"
run_tapt "klue/bert-base" $AEDA_DATASET $AEDA_REVISION $TAPT_EPOCHS "bert_on_aeda"
run_tapt "beomi/kcbert-base" $NIKL_DATASET $NIKL_REVISION $TAPT_EPOCHS "beomi-kcbert_on_nikl"
run_tapt "beomi/kcbert-base" $AEDA_DATASET $AEDA_REVISION $TAPT_EPOCHS "beomi-kcbert_on_aeda"

echo "--- 모든 TAPT 프로세스 완료 ---"

# ======================================================
# PHASE 2: Fine-tuning 시작 (순차 실행 및 결과 검증)
# ======================================================
echo "PHASE 2: Fine-tuning 시작"

# 기본 모델 5종 학습
run_finetune "klue/bert-base" "ft_klue-bert-base" "3e-5" "klue-bert-base"
run_finetune "beomi/kcbert-base" "ft_beomi-kcbert-base" "5e-5" "beomi-kcbert-base"
run_finetune "beomi/KcELECTRA-base-v2022" "ft_KcELECTRA" "5e-5" "KcELECTRA"
run_finetune "monologg/koelectra-small-v3-discriminator" "ft_koelectra-small" "5e-5" "koelectra-small"
run_finetune "monologg/koelectra-base-v3-discriminator" "ft_koelectra-base" "5e-5" "koelectra-base"

# TAPT된 모델 4종 학습
run_finetune "$TAPT_MODEL_DIR/bert_on_nikl" "ft_tapt-bert-nikl" "2e-5" "tapt-bert-nikl"
run_finetune "$TAPT_MODEL_DIR/beomi-kcbert_on_nikl" "ft_tapt-beomi-kcbert-nikl" "3e-5" "tapt-beomi-kcbert-nikl"
run_finetune "$TAPT_MODEL_DIR/bert_on_aeda" "ft_tapt-bert-aeda" "2e-5" "tapt-bert-aeda"
run_finetune "$TAPT_MODEL_DIR/beomi-kcbert_on_aeda" "ft_tapt-beomi-kcbert-aeda" "3e-5" "tapt-beomi-kcbert-aeda"

echo "--- 모든 Fine-tuning 프로세스 완료 ---"

# ======================================================
# PHASE 3: Ensemble Inference 시작
# ======================================================
echo "PHASE 3: Ensemble Inference 시작"

# 앙상블에 사용할 모델 경로 목록 자동 탐색
FT_BERT_PATH=$(find "$FT_MODEL_DIR/klue-bert-base/results" -type d -name "checkpoint-*" | head -n 1)
FT_BEOMI_KCBERT_PATH=$(find "$FT_MODEL_DIR/beomi-kcbert-base/results" -type d -name "checkpoint-*" | head -n 1)
FT_ELECTRA_PATH=$(find "$FT_MODEL_DIR/KcELECTRA/results" -type d -name "checkpoint-*" | head -n 1)
FT_KOELECTRA_SMALL_PATH=$(find "$FT_MODEL_DIR/koelectra-small/results" -type d -name "checkpoint-*" | head -n 1)
FT_KOELECTRA_BASE_PATH=$(find "$FT_MODEL_DIR/koelectra-base/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_BERT_NIKL_FT_PATH=$(find "$FT_MODEL_DIR/tapt-bert-nikl/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_BEOMI_KCBERT_NIKL_FT_PATH=$(find "$FT_MODEL_DIR/tapt-beomi-kcbert-nikl/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_BERT_AEDA_FT_PATH=$(find "$FT_MODEL_DIR/tapt-bert-aeda/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_BEOMI_KCBERT_AEDA_FT_PATH=$(find "$FT_MODEL_DIR/tapt-beomi-kcbert-aeda/results" -type d -name "checkpoint-*" | head -n 1)

# 전략 1: 기본 모델 5종 앙상블
echo "-> 전략 1: 기본 모델 5종 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_baseline_5models.csv" \
    --model_paths "$FT_BERT_PATH" "$FT_BEOMI_KCBERT_PATH" "$FT_ELECTRA_PATH" "$FT_KOELECTRA_SMALL_PATH" "$FT_KOELECTRA_BASE_PATH"

# 전략 2: TAPT (NIKL) 적용 모델 앙상블
echo "-> 전략 2: TAPT(NIKL) 적용 모델 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_nikl.csv" \
    --model_paths "$TAPT_BERT_NIKL_FT_PATH" "$TAPT_BEOMI_KCBERT_NIKL_FT_PATH" "$FT_KOELECTRA_BASE_PATH" "$FT_ELECTRA_PATH"

# 전략 3: TAPT (AEDA) 적용 모델 앙상블
echo "-> 전략 3: TAPT(AEDA) 적용 모델 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_aeda.csv" \
    --model_paths "$TAPT_BERT_AEDA_FT_PATH" "$TAPT_BEOMI_KCBERT_AEDA_FT_PATH" "$FT_KOELECTRA_BASE_PATH" "$FT_ELECTRA_PATH"

echo "======================================================"
echo "모든 워크플로우 완료!"
echo "생성된 최종 예측 파일:"
echo "- prediction_ensemble_baseline_5models.csv"
echo "- prediction_ensemble_tapt_nikl.csv"
echo "- prediction_ensemble_tapt_aeda.csv"
echo "======================================================"

