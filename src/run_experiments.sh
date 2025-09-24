#!/bin/bash

# --- 공통 설정 ---
MODEL_NAME="beomi/kcbert-base"
EPOCHS=30
BATCH_SIZE=64
GRAD_ACCUM_STEPS=2
WARMUP_STEPS=500

# ====================================================================================
# 실험 1: 학습률 (Learning Rate) 비교 (가장 일반적인 Weight Decay 값 고정)
# ====================================================================================
echo "===== 실험 1: 학습률 비교 시작 ====="

# 실험 1-1: Learning Rate = 3e-5 (안정적인 기준선)
nohup python main.py \
    --model_name "$MODEL_NAME" \
    --run_name "exp1_lr-3e-5_wd-0.01" \
    --epochs $EPOCHS \
    --lr 3e-5 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay 0.01 > logs/exp1_lr-3e-5.log 2>&1 &
echo "  - 실험 1-1 (lr=3e-5) 시작. 로그: logs/exp1_lr-3e-5.log"

# 실험 1-2: Learning Rate = 5e-5 (SOTA에서 가장 많이 사용되는 공격적인 값)
nohup python main.py \
    --model_name "$MODEL_NAME" \
    --run_name "exp1_lr-5e-5_wd-0.01" \
    --epochs $EPOCHS \
    --lr 5e-5 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay 0.01 > logs/exp1_lr-5e-5.log 2>&1 &
echo "  - 실험 1-2 (lr=5e-5) 시작. 로그: logs/exp1_lr-5e-5.log"

# 실험 1-3: Learning Rate = 2e-6
nohup python main.py \
    --model_name "$MODEL_NAME" \
    --run_name "exp1_lr-2e-6_wd-0.01" \
    --epochs $EPOCHS \
    --lr 2e-6 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay 0.01 > logs/exp1_lr-7e-5.log 2>&1 &
echo "  - 실험 1-3 (lr=2e-6) 시작. 로그: logs/exp1_lr-2e-6.log"


# ====================================================================================
# 실험 2: 가중치 감소 (Weight Decay) 비교 (실험 1에서 가장 좋았던 학습률 사용)
# 아래 lr 값은 실험 1의 wandb 결과를 보고 가장 좋았던 값으로 수정하세요. (예: 5e-5)
# ====================================================================================
echo ""
echo "===== 실험 2: Weight Decay 비교 시작 (최적 lr 사용) ====="
BEST_LR=5e-5

# 실험 2-1: Weight Decay = 0 (정규화 없음)
nohup python main.py \
    --model_name "$MODEL_NAME" \
    --run_name "exp2_lr-${BEST_LR}_wd-0" \
    --epochs $EPOCHS \
    --lr $BEST_LR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay 0 > logs/exp2_wd-0.log 2>&1 &
echo "  - 실험 2-1 (wd=0) 시작. 로그: logs/exp2_wd-0.log"

# 실험 2-2: Weight Decay = 0.1 (과적합 방지를 위해 조금 더 강한 정규화)
nohup python main.py \
    --model_name "$MODEL_NAME" \
    --run_name "exp2_lr-${BEST_LR}_wd-0.1" \
    --epochs $EPOCHS \
    --lr $BEST_LR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay 0.1 > logs/exp2_wd-0.1.log 2>&1 &
echo "  - 실험 2-2 (wd=0.1) 시작. 로그: logs/exp2_wd-0.1.log"

echo ""
echo "모든 실험이 백그라운드에서 시작되었습니다."
echo "wandb 대시보드에서 각 run_name을 확인하여 성능을 비교하세요."
