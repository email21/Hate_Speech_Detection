#!/binbash
# TAPT와 Fine-tuning이 완료된 4개의 모델 각각에 대한 추론을 실행하는 스크립트.
# 각 추론 결과는 별도의 파일로 저장됩니다.

# --- 환경 변수 설정 ---
# 추론에 사용할 데이터셋 정보
TARGET_DATASET="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
TARGET_REVISION="v1.2"

# Fine-tuning된 모델들이 저장된 기본 경로
FT_MODEL_DIR="../ft_models"

# 결과물을 저장할 폴더 생성
mkdir -p ./prediction

# --- 함수 정의 ---
# 지정된 모델로 추론을 실행하고 결과 파일을 재명명하는 함수
run_single_inference() {
    local model_dir_name=$1 # ex: "tapt-bert-nikl"
    local output_filename=$2  # ex: "prediction_tapt-bert-nikl.csv"
    
    # Fine-tuning된 모델의 최적 체크포인트 경로를 자동으로 찾습니다.
    local model_checkpoint_path=$(find "$FT_MODEL_DIR/$model_dir_name/results" -type d -name "checkpoint-*" | head -n 1)

    # 체크포인트가 존재하는지 확인합니다.
    if [ -z "$model_checkpoint_path" ]; then
        echo "오류: '$model_dir_name' 모델의 체크포인트를 찾을 수 없습니다. Fine-tuning이 성공적으로 완료되었는지 확인하세요."
        return 1
    fi
    
    echo "--- 추론 시작: $model_dir_name ---"
    
    # inference.py를 실행합니다.
    python inference.py \
        --dataset_name "$TARGET_DATASET" \
        --dataset_revision "$TARGET_REVISION" \
        --model_dir "$model_checkpoint_path"
        
    # 생성된 결과 파일의 이름을 지정된 이름으로 변경합니다.
    mv ./prediction/tapt_result.csv "./prediction/$output_filename"
    
    echo "--- 완료. 결과 저장: ./prediction/$output_filename ---"
}

# ===================================================================
# NIKL 데이터셋으로 TAPT 및 Fine-tuning된 모델 추론
# ===================================================================
run_single_inference "tapt-bert-nikl" "prediction_tapt-bert-nikl.csv"
run_single_inference "tapt-beomi-kcbert-nikl" "prediction_tapt-beomi-kcbert-nikl.csv"

# ===================================================================
# AEDA 데이터셋으로 TAPT 및 Fine-tuning된 모델 추론
# ===================================================================
run_single_inference "tapt-bert-aeda" "prediction_tapt-bert-aeda.csv"
run_single_inference "tapt-beomi-kcbert-aeda" "prediction_tapt-beomi-kcbert-aeda.csv"

echo "모든 개별 모델 추론이 완료되었습니다."
