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
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def predict_probabilities(model, dataloader, device):
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


def find_best_checkpoint(model_base_path):
    """주어진 경로 또는 하위 폴더에서 가장 마지막 체크포인트 폴더를 자동으로 찾습니다."""
    if not os.path.exists(model_base_path):
        print(f"Warning: '{model_base_path}' 폴더가 없습니다!")
        return None

    search_paths = [model_base_path]
    if os.path.exists(os.path.join(model_base_path, "results")):
        search_paths.append(os.path.join(model_base_path, "results"))

    best_checkpoint_path = None
    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue
        checkpoint_dirs = [
            d for d in os.listdir(search_path) if d.startswith("checkpoint-")
        ]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            best_checkpoint_path = os.path.join(search_path, checkpoint_dirs[-1])
            break  # 찾으면 더 이상 탐색하지 않음

    if best_checkpoint_path:
        print(
            f"Found best checkpoint for '{os.path.basename(model_base_path)}': '{os.path.basename(best_checkpoint_path)}'"
        )
    else:
        print(f"Warning: '{model_base_path}'에서 체크포인트를 찾지 못했습니다.")

    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Soft Voting Ensemble Script")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../NIKL_AU_2023_COMPETITION_v1.0",
        help="테스트 데이터셋이 있는 로컬 경로",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./prediction/ensemble_result.csv",
        help="최종 결과 파일명",
    )
    parser.add_argument("--max_len", type=int, default=256, help="최대 토큰 길이")
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="앙상블할 모델들의 기본 폴더 경로 리스트",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_paths = [find_best_checkpoint(p) for p in args.model_paths]
    best_model_paths = [p for p in best_model_paths if p is not None]

    if not best_model_paths:
        raise ValueError(
            "학습된 모델을 찾을 수 없습니다! 경로를 확인하거나 학습을 먼저 실행하세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(best_model_paths[0])
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

    all_model_probs = []
    for path in best_model_paths:
        print(f"--- Predicting with model from: {path} ---")
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
        probs = predict_probabilities(model, test_dataloader, device)
        all_model_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    print("--- Averaging probabilities (Soft Voting) ---")
    avg_probs = np.mean(all_model_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=-1)

    output = pd.DataFrame(
        {"id": test_df["id"], "input": test_df["input"], "output": final_preds}
    )
    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
    output.to_csv(args.output_filename, index=False)
    print(f"Ensemble prediction saved to {args.output_filename}")


if __name__ == "__main__":
    main()
