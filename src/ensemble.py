# ensemble.py

import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from data import construct_tokenized_dataset, hate_dataset
from datasets import load_dataset


def inference_for_ensemble(model, tokenized_sent, device, batch_size=32):
    """
    주어진 모델과 토큰화된 입력을 이용해 예측을 수행하는 함수
    """
    dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
    model.eval()
    output_logits = []

    # 배치 단위로 추론 수행
    for data in tqdm(dataloader, desc="Inferencing"):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        logits = outputs.logits.detach().cpu().numpy()
        output_logits.append(logits)

    # 모든 배치 결과를 하나의 배열로 결합
    return np.vstack(output_logits)


def run_ensemble(args):
    """
    여러 개의 모델을 불러와 테스트 데이터셋에 대해 앙상블 추론을 수행하는 함수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 테스트 데이터셋 불러오기
    print(
        f"--- Loading test dataset: {args.dataset_name} (revision: {args.dataset_revision}) ---"
    )
    test_df = load_dataset(
        args.dataset_name, split="test", revision=args.dataset_revision
    ).to_pandas()

    all_logits = []
    for model_path in args.model_paths:
        if not model_path or not model_path.strip():
            continue  # 경로가 비어있으면 건너뛰기
        print(f"\n--- Processing model from: {model_path} ---")

        # 모델과 토크나이저 불러오기
        config = AutoConfig.from_pretrained(model_path)
        original_model_name = config._name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_name, revision=args.model_revision
        )

        # 추가 토큰 등록
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

        # 데이터셋 토큰화 및 변환
        tokenized_test = construct_tokenized_dataset(
            test_df, tokenizer, args.max_len, original_model_name
        )
        test_dataloader = hate_dataset(tokenized_test, [0] * len(test_df))

        # 분류 모델 불러오기 및 토큰 임베딩 크기 조정
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            device
        )
        model.resize_token_embeddings(len(tokenizer))

        # 모델 추론 결과 저장
        logits = inference_for_ensemble(model, test_dataloader, device)
        all_logits.append(logits)

    # 앙상블 수행 불가 시 종료
    if not all_logits:
        print(
            "오류: 앙상블을 위한 유효한 모델이 하나도 없습니다. Phase 2의 학습 로그를 확인해주세요."
        )
        return

    # 소프트 보팅 방식으로 예측 결과 결합
    print(f"\n--- Performing Soft Voting Ensemble for {args.output_filename} ---")
    averaged_logits = np.mean(all_logits, axis=0)
    final_predictions = np.argmax(averaged_logits, axis=-1)

    # 최종 결과 저장
    output = pd.DataFrame(
        {"id": test_df["id"], "input": test_df["input"], "output": final_predictions}
    )
    output.to_csv(args.output_filename, index=False)
    print(f"--- Ensemble prediction saved to {args.output_filename} ---")


if __name__ == "__main__":
    # 명령행 인자 정의
    parser = argparse.ArgumentParser(description="Ensemble Inference Script")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument(
        "--output_filename", type=str, required=True, help="최종 제출 파일 이름"
    )
    args = parser.parse_args()

    # 모델 경로가 하나도 없을 경우 오류 발생
    if len(args.model_paths) < 1:
        raise ValueError(
            "앙상블을 위해서는 최소 1개 이상의 유효한 모델 경로가 필요합니다."
        )

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
