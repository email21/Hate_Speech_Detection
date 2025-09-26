# ensemble_hard_vote.py

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import mode
from tqdm import tqdm

# 기존 프로젝트의 데이터셋 클래스를 그대로 사용합니다.
from data import load_data, construct_tokenized_dataset, hate_dataset


def inference_single_model(model, dataloader, device):
    """단일 모델에 대한 추론을 수행하고 예측 라벨(numpy array)을 반환합니다."""
    model.eval()
    model.to(device)

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)

    return np.array(all_predictions)


def run_hard_voting_ensemble(args):
    """
    여러 모델 경로를 받아 하드 보팅 앙상블을 수행하고 결과를 저장합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 테스트 데이터셋 로드 및 준비
    print("--- Loading and preparing test dataset ---")
    test_df = load_data(args.dataset_name, "test", args.dataset_revision)
    # 임시 토크나이저 (데이터셋 준비용, 어떤 모델의 것이든 상관없음)
    temp_tokenizer = AutoTokenizer.from_pretrained(args.model_paths[0])

    tokenized_test = construct_tokenized_dataset(
        test_df, temp_tokenizer, args.max_len, args.model_paths[0]
    )
    # 라벨이 없는 테스트 데이터이므로 임시 라벨(0)을 넣어줍니다.
    dummy_labels = [0] * len(test_df)
    test_dataset = hate_dataset(tokenized_test, dummy_labels)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 2. 각 모델별로 추론 실행 및 결과 저장
    all_model_predictions = []

    for i, model_path in enumerate(args.model_paths):
        print(
            f"\n--- [{i+1}/{len(args.model_paths)}] Loading model from: {model_path} ---"
        )
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            predictions = inference_single_model(model, test_dataloader, device)
            all_model_predictions.append(predictions)
            print(f"--- Inference complete for model {i+1} ---")

        except Exception as e:
            print(
                f"Error loading or inferencing with model {model_path}. Skipping. Error: {e}"
            )
            continue

    if not all_model_predictions:
        print("Error: No valid predictions were generated. Exiting.")
        return

    # 3. 하드 보팅(Hard Voting) 수행
    print("\n--- Performing Hard Voting ---")
    # (모델 수, 데이터 수) 형태의 2D 배열로 변환
    predictions_array = np.array(all_model_predictions)

    # 각 데이터 샘플(column)에 대해 최빈값을 찾아 최종 예측값으로 결정 (axis=0)
    final_predictions, _ = mode(predictions_array, axis=0, keepdims=False)

    # 4. 최종 결과 파일 저장
    print(f"--- Saving final ensemble predictions to {args.output_filename} ---")
    output_df = pd.DataFrame(
        {"id": test_df["id"], "input": test_df["input"], "output": final_predictions}
    )

    output_dir = os.path.dirname(args.output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_df.to_csv(args.output_filename, index=False)
    print("--- Ensemble process complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard Voting Ensemble Script")
    parser.add_argument(
        "--dataset_name", type=str, default="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0"
    )
    parser.add_argument("--dataset_revision", type=str, default="v1.2")
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="List of model checkpoint paths for ensemble.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="Filename for the final submission CSV.",
    )
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_hard_voting_ensemble(args)
