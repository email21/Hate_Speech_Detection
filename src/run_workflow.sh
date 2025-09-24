#!/bin/bash
# TAPT 및 앙상블 자동화 워크플로우 스크립트
set -e

# --- 환경 변수 설정 ---
NIKL_DATASET="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
NIKL_REVISION="v1.2"
AEDA_DATASET="ensemble-2/AEDA-dataset"
AEDA_REVISION="v1.1"
TAPT_EPOCHS=5
GRAD_ACCUM_STEPS=2
TAPT_MODEL_DIR="../tapt_models"
FT_MODEL_DIR="../ft_models"
LOG_DIR="./logs"

# --- 초기화 ---
mkdir -p $TAPT_MODEL_DIR $FT_MODEL_DIR $LOG_DIR

# --- 함수 정의 ---
run_tapt() {
    local model_name=$1
    local dataset_name=$2
    local dataset_revision=$3
    local output_path="$TAPT_MODEL_DIR/$4"
    local log_file="$LOG_DIR/tapt_$4.log"

    echo "-> TAPT 시작: $model_name on $(basename $dataset_name)"
    python tapt.py --base_model_name "$model_name" --dataset_name "$dataset_name" --dataset_revision "$dataset_revision" --epochs $TAPT_EPOCHS --output_model_path "$output_path" > "$log_file" 2>&1
    
    if [ ! -f "$output_path/pytorch_model.bin" ] && [ ! -f "$output_path/model.safetensors" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: TAPT 실패! '$output_path'에 모델 파일이 생성되지 않았습니다."
        echo "로그 확인: $log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    echo "-> TAPT 완료: $4"
}

run_finetune() {
    local model_name=$1
    local run_name=$2
    local lr=$3
    local save_path_suffix=$4
    local save_path="$FT_MODEL_DIR/$save_path_suffix"
    local log_file="$LOG_DIR/$run_name.log"

    echo "-> Fine-tuning 시작: $run_name"
    python main.py --model_name "$model_name" --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --run_name "$run_name" --lr "$lr" --gradient_accumulation_steps $GRAD_ACCUM_STEPS --save_path "$save_path" > "$log_file" 2>&1
    
    if [ ! -d "$save_path/results" ] || [ -z "$(ls -A "$save_path/results")" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: Fine-tuning 실패! '$save_path/results' 디렉토리가 비어있습니다."
        echo "로그 확인: $log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    echo "-> Fine-tuning 완료: $run_name"
}

find_checkpoint() {
    local path=$(find "$FT_MODEL_DIR/$1/results" -type d -name "checkpoint-*" | head -n 1)
    if [ -z "$path" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: '$1' 모델의 체크포인트를 찾을 수 없습니다. Fine-tuning이 실패했는지 확인하세요."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    echo $path
}

# --- 워크플로우 실행 ---
echo "======================================================"
echo "PHASE 1: TAPT 시작 (순차 실행)"
echo "======================================================"
run_tapt "klue/bert-base" $NIKL_DATASET $NIKL_REVISION "bert_on_nikl"
run_tapt "klue/bert-base" $AEDA_DATASET $AEDA_REVISION "bert_on_aeda"
run_tapt "klue/kcbert-base" $NIKL_DATASET $NIKL_REVISION "kcbert_on_nikl"
run_tapt "klue/kcbert-base" $AEDA_DATASET $AEDA_REVISION "kcbert_on_aeda"
echo "--- 모든 TAPT 프로세스 완료 ---"

echo "======================================================"
echo "PHASE 2: Fine-tuning 시작 (순차 실행 및 결과 검증)"
echo "======================================================"
run_finetune "klue/bert-base" "ft_klue-bert-base" "3e-5" "klue-bert-base"
run_finetune "klue/kcbert-base" "ft_klue-kcbert-base" "5e-5" "klue-kcbert-base"
run_finetune "beomi/kcbert-base" "ft_beomi-kcbert-base" "5e-5" "beomi-kcbert-base"
run_finetune "beomi/KcELECTRA-base-v2022" "ft_KcELECTRA" "5e-5" "KcELECTRA"
run_finetune "monologg/koelectra-small-v3-discriminator" "ft_koelectra-small" "5e-5" "koelectra-small"
run_finetune "monologg/koelectra-base-v3-discriminator" "ft_koelectra-base" "5e-5" "koelectra-base"
run_finetune "$TAPT_MODEL_DIR/bert_on_nikl" "ft_tapt-bert-nikl" "3e-5" "tapt-bert-nikl"
run_finetune "$TAPT_MODEL_DIR/kcbert_on_nikl" "ft_tapt-kcbert-nikl" "5e-5" "tapt-kcbert-nikl"
run_finetune "$TAPT_MODEL_DIR/bert_on_aeda" "ft_tapt-bert-aeda" "3e-5" "tapt-bert-aeda"
run_finetune "$TAPT_MODEL_DIR/kcbert_on_aeda" "ft_tapt-kcbert-aeda" "5e-5" "tapt-kcbert-aeda"
echo "--- 모든 Fine-tuning 프로세스 완료 ---"

echo "======================================================"
echo "PHASE 3: Ensemble Inference 시작"
echo "======================================================"
FT_BERT_PATH=$(find_checkpoint "klue-bert-base")
FT_KCBERT_PATH=$(find_checkpoint "klue-kcbert-base")
FT_BEOMI_PATH=$(find_checkpoint "beomi-kcbert-base")
FT_ELECTRA_PATH=$(find_checkpoint "KcELECTRA")
FT_KOELECTRA_SMALL_PATH=$(find_checkpoint "koelectra-small")
FT_KOELECTRA_BASE_PATH=$(find_checkpoint "koelectra-base")
TAPT_BERT_NIKL_FT_PATH=$(find_checkpoint "tapt-bert-nikl")
TAPT_KCBERT_NIKL_FT_PATH=$(find_checkpoint "tapt-kcbert-nikl")
TAPT_BERT_AEDA_FT_PATH=$(find_checkpoint "tapt-bert-aeda")
TAPT_KCBERT_AEDA_FT_PATH=$(find_checkpoint "tapt-kcbert-aeda")

echo "-> 전략 1: 기본 모델 6종 앙상블"
python ensemble.py --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --output_filename "prediction_ensemble_baseline_6models.csv" --model_paths $FT_BERT_PATH $FT_KCBERT_PATH $FT_BEOMI_PATH $FT_ELECTRA_PATH $FT_KOELECTRA_SMALL_PATH $FT_KOELECTRA_BASE_PATH

echo "-> 전략 2: TAPT (NIKL) 적용 모델 앙상블"
python ensemble.py --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --output_filename "prediction_ensemble_tapt_nikl.csv" --model_paths $TAPT_BERT_NIKL_FT_PATH $TAPT_KCBERT_NIKL_FT_PATH $FT_KOELECTRA_BASE_PATH $FT_ELECTRA_PATH

echo "-> 전략 3: TAPT (AEDA) 적용 모델 앙상블"
python ensemble.py --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --output_filename "prediction_ensemble_tapt_aeda.csv" --model_paths $TAPT_BERT_AEDA_FT_PATH $TAPT_KCBERT_AEDA_FT_PATH $FT_KOELECTRA_BASE_PATH $FT_ELECTRA_PATH

echo "======================================================"
echo "모든 워크플로우 완료!"
echo "생성된 최종 예측 파일:"
echo "- prediction_ensemble_baseline_6models.csv"
echo "- prediction_ensemble_tapt_nikl.csv"
echo "- prediction_ensemble_tapt_aeda.csv"
echo "======================================================"