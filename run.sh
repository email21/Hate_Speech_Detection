#!/bin/bash

# =================================================================
# RoBERTa 기반 2-Phase 훈련 및 추론 파이프라인
#
# 사용법: 
# 1. 실행 권한을 부여: chmod +x run.sh
# 2. 스크립트 실행:  bash run.sh
# =================================================================
# python inference.py --model_name "./best_model" --model_dir "./final_best_model_0925" 
# --dataset_name "sagittarius5/NIKL_AU_2023_COMPETITION_v1.0" --batch_size 16
#     --dataset_name "$DATASET_NAME" \
#     --batch_size $BATCH_SIZE
# --- 변수 설정 ---
# 사용할 기본 모델 (Hugging Face Hub)
export BASE_MODEL="klue/roberta-large"

# 대회 데이터셋 이름 (Hugging Face Hub)
export DATASET_NAME="sagittarius5/NIKL_AU_2023_COMPETITION_v1.0"

# CPT를 거친 '도메인 전문가' 모델이 저장될 경로
export CPT_MODEL_DIR="./best_model"

# 최종 파인튜닝된 모델이 저장될 경로
export FINAL_MODEL_DIR="./final_best_model_0925"
s
# 배치 사이즈 (GPU VRAM에 맞춰 조절)
export BATCH_SIZE=16

echo "================================================="
echo "Phase 1: 추가 사전 훈련(CPT)을 시작합니다."
echo "Base Model: $BASE_MODEL"
echo "Output Dir: $CPT_MODEL_DIR"
echo "================================================="

# # pretrain.py 스크립트 실행
# python pretrain.py \
#     --model_name "$BASE_MODEL" \
#     --dataset_name "$DATASET_NAME" \
#     --model_dir "$CPT_MODEL_DIR" \
#     --batch_size $BATCH_SIZE \
#     --max_len 512 \
#     --cpt_epochs 3

echo "================================================="
echo "Phase 2: 파인튜닝을 시작합니다."
echo "CPT Model: $CPT_MODEL_DIR"
echo "Output Dir: $FINAL_MODEL_DIR"
echo "================================================="

# main.py 스크립트 실행
# --model_name에 CPT 모델 경로를 전달하는 것이 핵심!
python main.py \
    --run_name "roberta-large-cpt-finetuned_floss" \
    --model_name "$CPT_MODEL_DIR" \
    --dataset_name "$DATASET_NAME" \
    --model_dir "$FINAL_MODEL_DIR" \
    --batch_size $BATCH_SIZE \
    --lr 2e-5 \
    --epochs 5 \
    --warmup_steps 200 \
    --eval_step 200 \
    --save_step 200

echo "================================================="
echo "Phase 3: 추론을 시작합니다."
echo "Final Model: $FINAL_MODEL_DIR"
echo "================================================="

# inference.py 스크립트 실행
python inference.py \
    --model_name "$CPT_MODEL_DIR" \
    --model_dir "$FINAL_MODEL_DIR" \
    --dataset_name "$DATASET_NAME" \
    --batch_size $BATCH_SIZE

echo "The end ------- prediction/result.csv 파일을 확인하세요."
