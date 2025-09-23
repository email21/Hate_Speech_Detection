# ensemble.py

import torch
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CustomTorchDataset(TorchDataset):
    """토크나이징된 데이터를 Tensor로 변환하는 Dataset 클래스"""
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # pt 형식의 텐서에서 바로 아이템을 가져오도록 수정
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def predict_probabilities(model, dataloader, device):
    """모델의 예측 확률(softmax)을 반환하는 함수"""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting Probabilities"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs, axis=0)

def load_best_checkpoints(base_model_path, n_splits):
    """각 fold의 최적 체크포인트를 자동으로 찾는 함수"""
    model_folders = []
    for i in range(n_splits):
        fold_path = os.path.join(base_model_path, f"fold_{i+1}")
        if not os.path.exists(fold_path):
            print(f"Warning: {fold_path} 폴더가 존재하지 않습니다!")
            continue
            
        checkpoint_dirs = [d for d in os.listdir(fold_path) if d.startswith("checkpoint-")]
        if not checkpoint_dirs:
            print(f"Warning: {fold_path}에 체크포인트 폴더가 없습니다!")
            continue
            
        # 체크포인트 번호로 정렬하여 가장 마지막(best) 체크포인트 선택
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        best_checkpoint_path = os.path.join(fold_path, checkpoint_dirs[-1])
        model_folders.append(best_checkpoint_path)
        print(f"Fold {i+1}: {checkpoint_dirs[-1]} 선택됨")
    
    return model_folders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./model", help="K-Fold 모델들이 저장된 기본 경로")
    parser.add_argument("--dataset_dir", type=str, default="./NIKL_AU_2023_COMPETITION_v1.0", help="테스트 데이터셋 경로")
    parser.add_argument("--n_splits", type=int, default=5, help="Fold 수")
    parser.add_argument("--max_len", type=int, default=256, help="최대 토큰 길이")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("--- 각 Fold에서 Best checkpoints 찾기 ---")
    model_paths = load_best_checkpoints(args.save_path, args.n_splits)
    if not model_paths:
        raise ValueError("사용 가능한 모델 폴더가 없습니다!")

    print("--- 데이터 준비 ---")
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    test_df = pd.read_csv(os.path.join(args.dataset_dir, "test.csv"))
    
    tokenized_test = tokenizer(
        list(test_df["input"]),
        return_tensors="pt",
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    test_dataset = CustomTorchDataset(tokenized_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_fold_probs = []
    for i, path in enumerate(model_paths):
        print(f"--- Predicting with Fold {i+1} model from: {path} ---")
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
        
        probs = predict_probabilities(model, test_dataloader, device)
        all_fold_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    print("--- 확률 평균 계산 (Soft Voting) ---")
    avg_probs = np.mean(all_fold_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=-1)

    print("--- 최종 결과 파일 생성 ---")
    output = pd.DataFrame({"id": test_df["id"], "input": test_df["input"], "output": final_preds})
    result_path = "./prediction/"
    os.makedirs(result_path, exist_ok=True)
    output.to_csv(os.path.join(result_path, "ensemble_result.csv"), index=False)
    print(f"Ensemble prediction saved to {os.path.join(result_path, 'ensemble_result.csv')}")

if __name__ == "__main__":
    main()