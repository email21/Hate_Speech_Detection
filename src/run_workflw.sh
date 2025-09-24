#!/bin/bash
# TAPT부터 앙상블까지 전 과정을 자동화하는 SOTA 워크플로우 스크립트
# 오류 발생 시 즉시 중단되도록 설정
set -e

# --- 변수 설정 ---
# 데이터셋 정보
NIKL_DATASET="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
NIKL_REVISION="v1.2"
AEDA_DATASET="ensemble-2/AEDA-dataset"
AEDA_REVISION="v1.1"

# TAPT 에폭 (사용자 요청: 5)
TAPT_EPOCHS=5

# 모델 저장 경로
TAPT_MODEL_DIR="../tapt_models"
FT_MODEL_DIR="../ft_models"
LOG_DIR="./logs"

# 폴더 생성
mkdir -p $TAPT_MODEL_DIR $FT_MODEL_DIR $LOG_DIR

echo "======================================================"
echo "PHASE 1: Task-Adaptive Pre-training (TAPT) 시작"
echo "======================================================"

# 1. klue/bert-base TAPT (NIKL 데이터셋)
python tapt.py --base_model_name "klue/bert-base" --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --epochs $TAPT_EPOCHS --output_model_path "$TAPT_MODEL_DIR/bert_on_nikl" > "$LOG_DIR/tapt_bert_on_nikl.log" 2>&1 &
echo "-> TAPT 시작: klue/bert-base on NIKL (로그: $LOG_DIR/tapt_bert_on_nikl.log)"

# 2. klue/kcbert-base TAPT (NIKL 데이터셋)
python tapt.py --base_model_name "klue/kcbert-base" --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION --epochs $TAPT_EPOCHS --output_model_path "$TAPT_MODEL_DIR/kcbert_on_nikl" > "$LOG_DIR/tapt_kcbert_on_nikl.log" 2>&1 &
echo "-> TAPT 시작: klue/kcbert-base on NIKL (로그: $LOG_DIR/tapt_kcbert_on_nikl.log)"

# 모든 백그라운드 TAPT 프로세스가 끝날 때까지 대기
wait
echo "--- 모든 TAPT 프로세스 완료 ---"


echo "======================================================"
echo "PHASE 2: Fine-tuning 시작 (NIKL 데이터셋으로 학습)"
echo "======================================================"

# 6개의 기본 모델을 최적 하이퍼파라미터로 학습
python main.py --model_name "klue/bert-base" --run_name "ft_klue-bert-base" --lr 3e-5 --save_path "$FT_MODEL_DIR/klue-bert-base" > "$LOG_DIR/ft_klue-bert.log" 2>&1 &
echo "-> Fine-tuning 시작: klue/bert-base"

python main.py --model_name "klue/kcbert-base" --run_name "ft_klue-kcbert-base" --lr 5e-5 --save_path "$FT_MODEL_DIR/klue-kcbert-base" > "$LOG_DIR/ft_klue-kcbert.log" 2>&1 &
echo "-> Fine-tuning 시작: klue/kcbert-base"

python main.py --model_name "beomi/kcbert-base" --run_name "ft_beomi-kcbert-base" --lr 5e-5 --save_path "$FT_MODEL_DIR/beomi-kcbert-base" > "$LOG_DIR/ft_beomi-kcbert.log" 2>&1 &
echo "-> Fine-tuning 시작: beomi/kcbert-base"

python main.py --model_name "beomi/KcELECTRA-base-v2022" --run_name "ft_KcELECTRA" --lr 5e-5 --save_path "$FT_MODEL_DIR/KcELECTRA" > "$LOG_DIR/ft_KcELECTRA.log" 2>&1 &
echo "-> Fine-tuning 시작: beomi/KcELECTRA-base-v2022"

python main.py --model_name "monologg/koelectra-small-v3-discriminator" --run_name "ft_koelectra-small" --lr 5e-5 --save_path "$FT_MODEL_DIR/koelectra-small" > "$LOG_DIR/ft_koelectra-small.log" 2>&1 &
echo "-> Fine-tuning 시작: monologg/koelectra-small-v3-discriminator"

python main.py --model_name "monologg/koelectra-base-v3-discriminator" --run_name "ft_koelectra-base" --lr 5e-5 --save_path "$FT_MODEL_DIR/koelectra-base" > "$LOG_DIR/ft_koelectra-base.log" 2>&1 &
echo "-> Fine-tuning 시작: monologg/koelectra-base-v3-discriminator"

# 모든 백그라운드 Fine-tuning 프로세스가 끝날 때까지 대기
wait
echo "--- 모든 Fine-tuning 프로세스 완료 ---"


echo "======================================================"
echo "PHASE 3: Ensemble Inference 시작"
echo "======================================================"

# 앙상블에 사용할 모델 경로 목록 자동 탐색
# load_best_model_at_end=True 옵션으로 저장된 최적 체크포인트 경로를 찾음
FT_BERT_PATH=$(find $FT_MODEL_DIR/klue-bert-base/results -type d -name "checkpoint-*" | head -n 1)
FT_KCBERT_PATH=$(find $FT_MODEL_DIR/klue-kcbert-base/results -type d -name "checkpoint-*" | head -n 1)
FT_BEOMI_PATH=$(find $FT_MODEL_DIR/beomi-kcbert-base/results -type d -name "checkpoint-*" | head -n 1)
FT_ELECTRA_PATH=$(find $FT_MODEL_DIR/KcELECTRA/results -type d -name "checkpoint-*" | head -n 1)
FT_KOELECTRA_SMALL_PATH=$(find $FT_MODEL_DIR/koelectra-small/results -type d -name "checkpoint-*" | head -n 1)
FT_KOELECTRA_BASE_PATH=$(find $FT_MODEL_DIR/koelectra-base/results -type d -name "checkpoint-*" | head -n 1)

# 전략 1: 기본 모델 6종 앙상블 (가장 강력한 Baseline)
echo "-> 전략 1: 기본 모델 6종 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_baseline_6models.csv" \
    --model_paths $FT_BERT_PATH $FT_KCBERT_PATH $FT_BEOMI_PATH $FT_ELECTRA_PATH $FT_KOELECTRA_SMALL_PATH $FT_KOELECTRA_BASE_PATH

# 전략 2: TAPT (NIKL) 적용 모델 앙상블 (TAPT 효과 검증)
echo "-> 전략 2를 위한 TAPT(NIKL) 모델 Fine-tuning 시작"
python main.py --model_name "$TAPT_MODEL_DIR/bert_on_nikl" --run_name "ft_tapt-bert-nikl" --lr 3e-5 --save_path "$FT_MODEL_DIR/tapt-bert-nikl" > "$LOG_DIR/ft_tapt_bert_nikl.log" 2>&1 &
python main.py --model_name "$TAPT_MODEL_DIR/kcbert_on_nikl" --run_name "ft_tapt-kcbert-nikl" --lr 5e-5 --save_path "$FT_MODEL_DIR/tapt-kcbert-nikl" > "$LOG_DIR/ft_tapt_kcbert_nikl.log" 2>&1 &
wait
echo "-> 전략 2: TAPT(NIKL) 적용 모델 앙상블"
TAPT_BERT_NIKL_FT_PATH=$(find $FT_MODEL_DIR/tapt-bert-nikl/results -type d -name "checkpoint-*" | head -n 1)
TAPT_KCBERT_NIKL_FT_PATH=$(find $FT_MODEL_DIR/tapt-kcbert-nikl/results -type d -name "checkpoint-*" | head -n 1)
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_tapt_nikl.csv" \
    --model_paths $TAPT_BERT_NIKL_FT_PATH $TAPT_KCBERT_NIKL_FT_PATH $FT_KOELECTRA_BASE_PATH $FT_ELECTRA_PATH # 성능 좋은 4개 조합

# 전략 3: 가장 성능이 좋을 것으로 예상되는 모델 3개 앙상블
echo "-> 전략 3: Top-3 모델 앙상블"
python ensemble.py \
    --dataset_name $NIKL_DATASET --dataset_revision $NIKL_REVISION \
    --output_filename "prediction_ensemble_top3.csv" \
    --model_paths $FT_KOELECTRA_BASE_PATH $FT_KCBERT_PATH $FT_BERT_PATH

echo "======================================================"
echo "최종 결과물:"
echo "- prediction_ensemble_baseline_6models.csv"
echo "- prediction_ensemble_tapt_nikl.csv"
echo "- prediction_ensemble_top3.csv"
echo "======================================================"
