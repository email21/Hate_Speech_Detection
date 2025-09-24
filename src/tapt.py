# tapt.py
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


def run_tapt(args):
    """지정된 모델을 대회 데이터로 TAPT를 수행합니다."""

    # 1. HuggingFace Hub에서 데이터셋 로드 (revision 적용)
    print(
        f"--- Loading dataset: {args.dataset_name} (revision: {args.dataset_revision}) ---"
    )
    dataset = load_dataset(
        args.dataset_name, split="train", revision=args.dataset_revision
    )

    # 'input' 컬럼만 텍스트 파일로 저장 (TAPT 학습용)
    with open("tapt_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["input"]))

    # 2. 기본 모델 및 토크나이저 로드 (revision 적용)
    print(
        f"--- Loading base model: {args.base_model_name} (revision: {args.model_revision}) ---"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, revision=args.model_revision
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.base_model_name, revision=args.model_revision
    )

    # 3. TAPT용 데이터셋 준비
    tapt_dataset = load_dataset("text", data_files={"train": "tapt_corpus.txt"})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )

    tokenized_dataset = tapt_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # 4. TAPT 학습 설정 및 실행
    training_args = TrainingArguments(
        output_dir=args.output_model_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="epoch",
        save_total_limit=1,
        prediction_loss_only=True,
        logging_steps=100,
        fp16=True,  # 혼합 정밀도 학습으로 속도 및 메모리 효율 향상
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=0.15
        ),
    )

    print(f"--- Starting TAPT for {args.base_model_name} on {args.dataset_name} ---")
    trainer.train()
    trainer.save_model(args.output_model_path)
    print(f"--- TAPT finished. Model saved to {args.output_model_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-Adaptive Pre-training Script")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Hugging Face 데이터셋 이름"
    )
    parser.add_argument(
        "--dataset_revision", type=str, default="main", help="데이터셋 버전"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="TAPT를 수행할 기본 모델 이름",
    )
    parser.add_argument("--model_revision", type=str, default="main", help="모델 버전")
    parser.add_argument(
        "--output_model_path", type=str, required=True, help="TAPT된 모델이 저장될 경로"
    )
    parser.add_argument("--epochs", type=int, default=3, help="TAPT 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="TAPT 배치 사이즈")
    args = parser.parse_args()

    run_tapt(args)


#### TAPT 실행 방법

# `klue/bert-base` 모델을 3 에폭 동안 TAPT하여 `./tapt_klue-bert-base` 폴더에 저장.

# 실행 방법 ==================
# python tapt.py \
#     --base_model_name "klue/bert-base" \
#     --output_model_path "./tapt_klue-bert-base" \
#     --epochs 3
