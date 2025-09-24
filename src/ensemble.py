# ensemble.py

import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from data import construct_tokenized_dataset, hate_dataset  # 기존 data.py의 함수 재사용
from datasets import load_dataset


def inference_for_ensemble(model, tokenized_sent, device, batch_size=32):
    """앙상블을 위해 로짓(logits)을 반환하는 추론 함수"""
    dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
    model.eval()
    output_logits = []

    for data in tqdm(dataloader, desc="Inferencing"):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        logits = outputs.logits.detach().cpu().numpy()
        output_logits.append(logits)

    return np.vstack(output_logits)


def run_ensemble(args):
    """지정된 모델 경로들로 앙상블을 수행합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 테스트 데이터셋 로드 (최초 1회만)
    print(
        f"--- Loading test dataset: {args.dataset_name} (revision: {args.dataset_revision}) ---"
    )
    test_df = load_dataset(
        args.dataset_name, split="test", revision=args.dataset_revision
    ).to_pandas()

    all_logits = []

    # 2. 각 모델을 순회하며 추론 및 로짓 수집
    for model_path in args.model_paths:
        print(f"\n--- Processing model from: {model_path} ---")

        # 저장된 config에서 원본 모델 이름(_name_or_path)을 가져옴
        config = AutoConfig.from_pretrained(model_path)
        original_model_name = config._name_or_path

        # 토크나이저 로드 (new_tokens 추가)
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_name, revision=args.model_revision
        )
        new_tokens = [
            "&name&",
            "&location&",
            "&affiliation&",
            "&company&",
            "&brand&",
            "&art&",
            "&other&",
            "&nama&",
            "&affifiation&",
            "&name",
            "&online-account&",
            "&compnay&",
            "&anme&",
            "& name&",
            "&address&",
            "&tel-num&",
            "&naem&",
        ]
        tokenizer.add_tokens(new_tokens)

        # 토크나이저를 사용해 데이터셋 준비
        tokenized_test = construct_tokenized_dataset(
            test_df, tokenizer, args.max_len, original_model_name
        )
        test_dataloader = hate_dataset(
            tokenized_test, [0] * len(test_df)
        )  # label은 임의값

        # 학습된 모델 로드
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            device
        )
        model.resize_token_embeddings(len(tokenizer))  # new_tokens 반영

        # 추론 실행
        logits = inference_for_ensemble(model, test_dataloader, device)
        all_logits.append(logits)

    # 3. Soft Voting: 수집된 로짓들의 평균 계산
    print("\n--- Performing Soft Voting Ensemble ---")
    averaged_logits = np.mean(all_logits, axis=0)
    final_predictions = np.argmax(averaged_logits, axis=-1)

    # 4. 제출 파일 생성
    output = pd.DataFrame(
        {"id": test_df["id"], "input": test_df["input"], "output": final_predictions}
    )
    save_path = "./prediction_ensemble.csv"
    output.to_csv(save_path, index=False)
    print(f"--- Ensemble prediction saved to {save_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Inference Script")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Hugging Face 데이터셋 이름"
    )
    parser.add_argument(
        "--dataset_revision", type=str, default="main", help="데이터셋 버전"
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="앙상블에 사용될 모델들의 원본 버전",
    )
    parser.add_argument("--max_len", type=int, default=256, help="최대 시퀀스 길이")
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="앙상블에 사용할 학습된 모델들의 경로 (최소 2개 이상)",
    )
    args = parser.parse_args()

    if len(args.model_paths) < 2:
        raise ValueError("앙상블을 위해서는 최소 2개 이상의 모델 경로가 필요합니다.")

    run_ensemble(args)

#### 앙상블 실행 방법

# 먼저, 5개의 모델을 각각 `main.py`로 학습시켜야 . 각 모델의 최적 체크포인트가 아래와 같은 경로에 저장되었다고 가정.

#   * `./models/roberta-large/checkpoint-best`
#   * `./models/koelectra-v3/checkpoint-best`
#   * `./models/kcelectra/checkpoint-best`
#   * `./models/kobert/checkpoint-best`
#   * `./models/klue-bert/checkpoint-best`

# 그 후, 아래 명령어로 앙상블 스크립트를 실행.

# python ensemble.py \
#     --model_paths \
#     ./models/roberta-large/checkpoint-best \
#     ./models/koelectra-v3/checkpoint-best \
#     ./models/kcelectra/checkpoint-best \
#     ./models/kobert/checkpoint-best \
#     ./models/klue-bert/checkpoint-best


# 이 명령어는 5개 모델의 예측 결과를 종합하여 최종 제출 파일 `prediction_ensemble.csv`을 생성. jsonl 파일 변환 후 대회 제출.
