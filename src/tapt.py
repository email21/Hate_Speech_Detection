# tapt.py
import argparse
import os
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
    corpus_file = "tapt_corpus.txt"
    try:
        print(
            f"--- Loading dataset: {args.dataset_name} (revision: {args.dataset_revision}) ---"
        )
        dataset = load_dataset(
            args.dataset_name, split="train", revision=args.dataset_revision
        )

        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        if "input" not in dataset.column_names:
            raise ValueError(
                f"Dataset does not have 'input' column. Available columns: {dataset.column_names}"
            )

        print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")

        with open(corpus_file, "w", encoding="utf-8") as f:
            for example in dataset:
                text = example["input"]
                if text and text.strip():
                    f.write(text.strip() + "\n")

        print(
            f"--- Loading base model: {args.base_model_name} (revision: {args.model_revision}) ---"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name, revision=args.model_revision
        )
        model = AutoModelForMaskedLM.from_pretrained(
            args.base_model_name, revision=args.model_revision
        )

        # --- Fine-tuning과 동일한 특수 토큰을 TAPT 전에 추가 ---
        print("--- Adding special tokens before TAPT ---")
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
        model.resize_token_embeddings(len(tokenizer))
        print(f"--- Tokenizer and model resized. New vocab size: {len(tokenizer)} ---")
        # -----------------------------------------------------------------

        tapt_dataset = load_dataset("text", data_files={"train": corpus_file})

        def tokenize_function(examples):
            texts = [text for text in examples["text"] if text and text.strip()]
            if not texts:
                return {"input_ids": [], "attention_mask": []}
            return tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=512,
                return_special_tokens_mask=True,
            )

        print("--- Tokenizing dataset ---")
        tokenized_dataset = tapt_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"], num_proc=4
        )

        training_args = TrainingArguments(
            output_dir=args.output_model_path,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            save_strategy="epoch",
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=100,
            fp16=True,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )

        print(
            f"--- Starting TAPT for {args.base_model_name} on {args.dataset_name} ---"
        )
        trainer.train()

        print(f"--- Saving model and tokenizer to {args.output_model_path} ---")
        trainer.save_model(args.output_model_path)
        tokenizer.save_pretrained(args.output_model_path)

        print(f"--- TAPT finished ---")

    except Exception as e:
        print(f"Error during TAPT: {str(e)}")
        raise e
    finally:
        # 임시 파일 정리
        if os.path.exists(corpus_file):
            os.remove(corpus_file)
            print(f"Cleaned up temporary file: {corpus_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-Adaptive Pre-training Script")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_revision", type=str, default="main", help="Dataset revision/branch"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="Base model name from HuggingFace",
    )
    parser.add_argument(
        "--model_revision", type=str, default="main", help="Model revision/branch"
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        required=True,
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size per device"
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("Epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    run_tapt(args)
