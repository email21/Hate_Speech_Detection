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

    print(
        f"--- Loading test dataset: {args.dataset_name} (revision: {args.dataset_revision}) ---"
    )
    test_df = load_dataset(
        args.dataset_name, split="test", revision=args.dataset_revision
    ).to_pandas()

    all_logits = []
    for model_path in args.model_paths:
        if not model_path or not model_path.strip():
            continue
        print(f"\n--- Processing model from: {model_path} ---")

        # --- [핵심 수정 로직 시작] ---
        # 1. 체크포인트 폴더의 config.json을 먼저 로드합니다.
        config = AutoConfig.from_pretrained(model_path)
        # 2. config에서 원본 모델 이름 (_name_or_path)을 가져옵니다.
        original_model_name = config._name_or_path

        print(f"    (Found original model name: '{original_model_name}')")

        # 3. 원본 모델 이름으로 Hub에서 정확한 토크나이저를 로드합니다.
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_name, revision=args.model_revision
        )
        # --- [핵심 수정 로직 종료] ---

        # 학습 때와 동일하게 특수 토큰 추가
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

        tokenized_test = construct_tokenized_dataset(
            test_df, tokenizer, args.max_len, original_model_name
        )
        test_dataloader = hate_dataset(tokenized_test, [0] * len(test_df))

        # 체크포인트 경로에서 학습된 모델 가중치를 로드
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            device
        )
        model.resize_token_embeddings(len(tokenizer))

        logits = inference_for_ensemble(model, test_dataloader, device)
        all_logits.append(logits)

    if not all_logits:
        print(
            "오류: 앙상블을 위한 유효한 모델이 하나도 없습니다. Phase 2의 학습 로그를 확인해주세요."
        )
        return

    print(f"\n--- Performing Soft Voting Ensemble for {args.output_filename} ---")
    averaged_logits = np.mean(all_logits, axis=0)
    final_predictions = np.argmax(averaged_logits, axis=-1)

    output = pd.DataFrame(
        {"id": test_df["id"], "input": test_df["input"], "output": final_predictions}
    )
    output.to_csv(args.output_filename, index=False)
    print(f"--- Ensemble prediction saved to {args.output_filename} ---")


if __name__ == "__main__":
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

    if len(args.model_paths) < 1:
        raise ValueError(
            "앙상블을 위해서는 최소 1개 이상의 유효한 모델 경로가 필요합니다."
        )

    run_ensemble(args)
