#!/bin/bash
# PHASE 3: Ensemble Inference (앙상블 추론)만 실행하는 스크립트
# 각 Fine-tuning 단계의 결과물이 존재하는지 확인 후 앙상블을 수행합니다.

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

# --- 변수 설정 ---
# 추론에 사용할 데이터셋 정보
NIKL_DATASET="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
NIKL_REVISION="v1.2"

# Fine-tuning된 모델들이 저장된 기본 경로
FT_MODEL_DIR="../ft_models"

echo "======================================================"
echo "PHASE 3: Ensemble Inference 시작"
echo "Fine-tuned 모델 경로: $FT_MODEL_DIR"
echo "======================================================"

# --- 함수 정의 ---
# 지정된 경로에서 최적 체크포인트를 찾는 함수
# 경로가 없으면 오류 메시지와 함께 스크립트를 중단시킴
find_checkpoint() {
    local model_dir_name=$1
    local full_path="$FT_MODEL_DIR/$model_dir_name/results"
    
    if [ ! -d "$full_path" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: Fine-tuning 결과 폴더를 찾을 수 없습니다: $full_path"
        echo "PHASE 2가 해당 모델에 대해 성공적으로 완료되었는지 확인하세요."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    
    local checkpoint_path=$(find "$full_path" -type d -name "checkpoint-*" | head -n 1)
    
    if [ -z "$checkpoint_path" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "오류: 체크포인트를 찾을 수 없습니다: $full_path"
        echo "모델 학습 로그를 확인하여 Fine-tuning이 성공했는지 확인하세요."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
    
    echo "$checkpoint_path"
}

# --- 앙상블에 필요한 모델 경로 탐색 ---
echo "--- 앙상블에 사용할 모델들의 최적 체크포인트 경로를 찾습니다... ---"
FT_BERT_PATH=$(find_checkpoint "klue-bert-base")
FT_BEOMI_KCBERT_PATH=$(find_checkpoint "beomi-kcbert-base")
FT_ELECTRA_PATH=$(find_checkpoint "KcELECTRA")
FT_KOELECTRA_SMALL_PATH=$(find_checkpoint "koelectra-small")
FT_KOELECTRA_BASE_PATH=$(find_checkpoint "koelectra-base")
TAPT_BERT_NIKL_FT_PATH=$(find_checkpoint "tapt-bert-nikl")
TAPT_BEOMI_KCBERT_NIKL_FT_PATH=$(find_checkpoint "tapt-beomi-kcbert-nikl")
TAPT_BERT_AEDA_FT_PATH=$(find_checkpoint "tapt-bert-aeda")
TAPT_BEOMI_KCBERT_AEDA_FT_PATH=$(find_checkpoint "tapt-beomi-kcbert-aeda")
echo "--- 모든 체크포인트 경로를 성공적으로 찾았습니다. ---"

# --- 앙상블 전략 실행 ---

# 전략 1: 기본 모델 5종 앙상블
echo -e "\n-> 전략 1: 기본 모델 5종 앙상블 실행"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_baseline_5models.csv" \
    --model_paths "$FT_BERT_PATH" "$FT_BEOMI_KCBERT_PATH" "$FT_ELECTRA_PATH" "$FT_KOELECTRA_SMALL_PATH" "$FT_KOELECTRA_BASE_PATH"

# 전략 2: TAPT (NIKL) 적용 모델 앙상블
echo -e "\n-> 전략 2: TAPT(NIKL) 적용 모델 앙상블 실행"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_nikl.csv" \
    --model_paths "$TAPT_BERT_NIKL_FT_PATH" "$TAPT_BEOMI_KCBERT_NIKL_FT_PATH" "$FT_KOELECTRA_BASE_PATH" "$FT_ELECTRA_PATH"

# 전략 3: TAPT (AEDA) 적용 모델 앙상블
echo -e "\n-> 전략 3: TAPT(AEDA) 적용 모델 앙상블 실행"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_aeda.csv" \
    --model_paths "$TAPT_BERT_AEDA_FT_PATH" "$TAPT_BEOMI_KCBERT_AEDA_FT_PATH" "$FT_KOELECTRA_BASE_PATH" "$FT_ELECTRA_PATH"

echo -e "\n======================================================"
echo "모든 앙상블 추론 완료!"
echo "생성된 최종 예측 파일:"
echo "- prediction_ensemble_baseline_5models.csv"
echo "- prediction_ensemble_tapt_nikl.csv"
echo "- prediction_ensemble_tapt_aeda.csv"
echo "======================================================"
