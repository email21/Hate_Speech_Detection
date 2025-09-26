import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

# 다른 .py 파일에서 필요한 함수들을 가져옵니다.
from arguments import add_common_args, add_infer_args
from data import prepare_dataset
from model import load_model_for_inference, inference_for_ensemble

def main(args):
    """
    K-Fold로 학습된 여러 모델의 예측을 앙상블하여 최종 결과를 도출합니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 모든 모델 폴더 경로를 자동으로 찾기
    model_base_dir = args.model_dir
    model_paths = [
        os.path.join(model_base_dir, dir_name) 
        for dir_name in os.listdir(model_base_dir) 
        if os.path.isdir(os.path.join(model_base_dir, dir_name))
    ]
    
    if not model_paths:
        print(f"오류: '{model_base_dir}' 디렉토리에서 모델 폴더를 찾을 수 없습니다.")
        return

    print(f"발견된 모델 폴더: {model_paths}")
    
    # 2. 테스트 데이터셋 준비 (첫번째 모델의 토크나이저 기준)
    temp_tokenizer, _ = load_model_for_inference(args.model_name, model_paths[0])
    
    # [수정된 부분] prepare_dataset 함수에 args.revision 인자를 추가로 전달합니다.
    _, _, hate_test_dataset, test_dataset_df = prepare_dataset(
        args.dataset_dir, temp_tokenizer, args.max_len, args.model_name, args.revision
    )

    all_logits = []
    print("\n--- Starting Ensemble Inference ---")

    # 3. 각 모델을 순회하며 로짓(logit) 예측
    for path in tqdm(model_paths, desc="Ensembling Models"):
        print(f"Loading model from: {path}")
        _, model = load_model_for_inference(args.model_name, path)
        model.to(device)
        
        logits = inference_for_ensemble(model, hate_test_dataset, device)
        all_logits.append(logits)

    # 4. 모든 로짓의 평균을 계산
    avg_logits = np.mean(all_logits, axis=0)

    # 5. 평균 로짓을 기반으로 최종 라벨 결정
    final_predictions = np.argmax(avg_logits, axis=-1).tolist()
    print("--- Ensemble Prediction Done ---")
    
    # 6. 결과 CSV 파일로 저장
    output = pd.DataFrame({
        "id": test_dataset_df["id"],
        "input": test_dataset_df["input"],
        "output": final_predictions,
    })

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, "result.csv")
    output.to_csv(result_path, index=False)
    print(f"--- Saved ensemble result to {result_path} ---")

# ensemble_inference.py 의 parse_args 함수 최종본

def parse_args():
    """
    앙상블 추론에 필요한 인자를 파싱하는 함수
    """
    parser = argparse.ArgumentParser(description="Ensemble Inference arguments")
    # arguments.py의 함수를 호출해 인자를 추가합니다.
    # add_common_args 에서 revision 인자도 처리됩니다.
    parser = add_common_args(parser)
    parser = add_infer_args(parser)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)