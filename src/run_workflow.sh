#!/bin/bash
# TAPT 및 앙상블 자동화 워크플로우 스크립트
# GPU 자원 부족 문제를 해결하기 위해 모든 학습을 순차적으로 실행하도록 수정

# 오류 발생 시 즉시 중단
set -e

# --- 변수 설정 ---
NIKL_DATASET="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
NIKL_REVISION="v1.2"
AEDA_DATASET="ensemble-2/AEDA-dataset"
AEDA_REVISION="v1.1"
TAPT_EPOCHS=5
TAPT_MODEL_DIR="../tapt_models"
FT_MODEL_DIR="../ft_models"
LOG_DIR="./logs"

# 폴더 생성
mkdir -p $TAPT_MODEL_DIR $FT_MODEL_DIR $LOG_DIR

# --- 함수 정의 ---
# 학습(Fine-tuning)을 실행하고 결과를 검증하는 함수
run_finetune() {
    local model_name=$1
    local run_name=$2
    local lr=$3
    local save_path="$FT_MODEL_DIR/$4"
    local log_file="$LOG_DIR/$2.log"

    echo "-> Fine-tuning 시작: $model_name"
    python main.py --model_name "$model_name" --run_name "$run_name" --lr "$lr" --save_path "$save_path" > "$log_file" 2>&1
    
    # [핵심] 결과 폴더 생성 여부 확인
    if [ ! -d "$save_path/results" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: $model_name 의 Fine-tuning 실패!"
        echo "'$save_path/results' 디렉토리가 생성되지 않았습니다."
        echo "자세한 원인은 로그 파일을 확인하세요: $log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    echo "-> Fine-tuning 완료: $model_name"
}

# ======================================================
# PHASE 1: TAPT 시작 (순차 실행)
# ======================================================
echo "PHASE 1: Task-Adaptive Pre-training (TAPT) 시작"
python tapt.py --base_model_name "klue/bert-base" --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --epochs $TAPT_EPOCHS --output_model_path "$TAPT_MODEL_DIR/bert_on_nikl" > "$LOG_DIR/tapt_bert_on_nikl.log" 2>&1
echo "-> TAPT 완료: klue/bert-base on NIKL"

python tapt.py --base_model_name "klue/bert-base" --dataset_name $AEDA_DATASET --dataset_revision $AEDA_REVISION --epochs $TAPT_EPOCHS --output_model_path "$TAPT_MODEL_DIR/bert_on_aeda" > "$LOG_DIR/tapt_bert_on_aeda.log" 2>&1
echo "-> TAPT 완료: klue/bert-base on AEDA"

python tapt.py --base_model_name "klue/kcbert-base" --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --epochs $TAPT_EPOCHS --output_model_path "$TAPT_MODEL_DIR/kcbert_on_nikl" > "$LOG_DIR/tapt_kcbert_on_nikl.log" 2>&1
echo "-> TAPT 완료: klue/kcbert-base on NIKL"

python tapt.py --base_model_name "klue/kcbert-base" --dataset_name $AEDA_DATASET --dataset_revision $AEDA_REVISION --epochs $TAPT_EPOCHS --output_model_path "$TAPT_MODEL_DIR/kcbert_on_aeda" > "$LOG_DIR/tapt_kcbert_on_aeda.log" 2>&1
echo "-> TAPT 완료: klue/kcbert-base on AEDA"

echo "--- 모든 TAPT 프로세스 완료 ---"

# ======================================================
# PHASE 2: Fine-tuning 시작 (순차 실행 및 결과 검증)
# ======================================================
echo "PHASE 2: Fine-tuning 시작"

run_finetune "klue/bert-base" "ft_klue-bert-base" "3e-5" "klue-bert-base"
run_finetune "klue/kcbert-base" "ft_klue-kcbert-base" "5e-5" "klue-kcbert-base"
run_finetune "beomi/kcbert-base" "ft_beomi-kcbert-base" "5e-5" "beomi-kcbert-base"
run_finetune "beomi/KcELECTRA-base-v2022" "ft_KcELECTRA" "5e-5" "KcELECTRA"
run_finetune "monologg/koelectra-small-v3-discriminator" "ft_koelectra-small" "5e-5" "koelectra-small"
run_finetune "monologg/koelectra-base-v3-discriminator" "ft_koelectra-base" "5e-5" "koelectra-base"

# TAPT된 모델 학습
run_finetune "$TAPT_MODEL_DIR/bert_on_nikl" "ft_tapt-bert-nikl" "3e-5" "tapt-bert-nikl"
run_finetune "$TAPT_MODEL_DIR/kcbert_on_nikl" "ft_tapt-kcbert-nikl" "5e-5" "tapt-kcbert-nikl"
run_finetune "$TAPT_MODEL_DIR/bert_on_aeda" "ft_tapt-bert-aeda" "3e-5" "tapt-bert-aeda"
run_finetune "$TAPT_MODEL_DIR/kcbert_on_aeda" "ft_tapt-kcbert-aeda" "5e-5" "tapt-kcbert-aeda"

echo "--- 모든 Fine-tuning 프로세스 완료 ---"


# ======================================================
# PHASE 3: Ensemble Inference 시작
# ======================================================
echo "PHASE 3: Ensemble Inference 시작"

# 앙상블에 사용할 모델 경로 목록 자동 탐색
FT_BERT_PATH=$(find "$FT_MODEL_DIR/klue-bert-base/results" -type d -name "checkpoint-*" | head -n 1)
FT_KCBERT_PATH=$(find "$FT_MODEL_DIR/klue-kcbert-base/results" -type d -name "checkpoint-*" | head -n 1)
FT_BEOMI_PATH=$(find "$FT_MODEL_DIR/beomi-kcbert-base/results" -type d -name "checkpoint-*" | head -n 1)
FT_ELECTRA_PATH=$(find "$FT_MODEL_DIR/KcELECTRA/results" -type d -name "checkpoint-*" | head -n 1)
FT_KOELECTRA_SMALL_PATH=$(find "$FT_MODEL_DIR/koelectra-small/results" -type d -name "checkpoint-*" | head -n 1)
FT_KOELECTRA_BASE_PATH=$(find "$FT_MODEL_DIR/koelectra-base/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_BERT_NIKL_FT_PATH=$(find "$FT_MODEL_DIR/tapt-bert-nikl/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_KCBERT_NIKL_FT_PATH=$(find "$FT_MODEL_DIR/tapt-kcbert-nikl/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_BERT_AEDA_FT_PATH=$(find "$FT_MODEL_DIR/tapt-bert-aeda/results" -type d -name "checkpoint-*" | head -n 1)
TAPT_KCBERT_AEDA_FT_PATH=$(find "$FT_MODEL_DIR/tapt-kcbert-aeda/results" -type d -name "checkpoint-*" | head -n 1)

# 전략 1: 기본 모델 6종 앙상블
echo "-> 전략 1: 기본 모델 6종 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_baseline_6models.csv" \
    --model_paths $FT_BERT_PATH $FT_KCBERT_PATH $FT_BEOMI_PATH $FT_ELECTRA_PATH $FT_KOELECTRA_SMALL_PATH $FT_KOELECTRA_BASE_PATH

# 전략 2: TAPT (NIKL) 적용 모델 앙상블
echo "-> 전략 2: TAPT(NIKL) 적용 모델 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_nikl.csv" \
    --model_paths $TAPT_BERT_NIKL_FT_PATH $TAPT_KCBERT_NIKL_FT_PATH $FT_KOELECTRA_BASE_PATH $FT_ELECTRA_PATH

# 전략 3: TAPT (AEDA) 적용 모델 앙상블
echo "-> 전략 3: TAPT(AEDA) 적용 모델 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_aeda.csv" \
    --model_paths $TAPT_BERT_AEDA_FT_PATH $TAPT_KCBERT_AEDA_FT_PATH $FT_KOELECTRA_BASE_PATH $FT_ELECTRA_PATH

echo "======================================================"
echo "모든 워크플로우 완료!"
echo "생성된 최종 예측 파일:"
echo "- prediction_ensemble_baseline_6models.csv"
echo "- prediction_ensemble_tapt_nikl.csv"
echo "- prediction_ensemble_tapt_aeda.csv"
echo "======================================================"