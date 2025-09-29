# main.py

import os
import argparse
import pandas as pd
import wandb
from sklearn.model_selection import StratifiedKFold
from model import train, load_tokenizer_and_model_for_train
from data import construct_tokenized_dataset, hate_dataset, load_data
from arguments import add_common_args, add_train_args


def parse_args():
    """학습에 사용되는 arguments를 관리"""
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    parser.add_argument(
        "--n_splits",
        type=int,
        default=0,
        help="K-Fold 분할 수. 0 또는 1이면 단일 학습 수행.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    if args.n_splits > 1:
        # --- K-Fold 학습 로직 ---
        print(f"--- Starting {args.n_splits}-Fold Cross Validation ---")

        # HuggingFace Hub에서 전체 학습 데이터를 한 번만 로드
        full_train_df = load_data(args.dataset_name, "train", args.dataset_revision)
        tokenizer, _ = load_tokenizer_and_model_for_train(args)  # 토크나이저만 로드

        skf = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=args.seed
        )
        os.makedirs(args.save_path, exist_ok=True)

        for fold, (train_indices, val_indices) in enumerate(
            skf.split(full_train_df, full_train_df["output"])
        ):
            print(f"=============== FOLD {fold+1}/{args.n_splits} ================")

            # 현재 Fold에 맞는 데이터 분할 및 토크나이징
            fold_train_df = full_train_df.iloc[train_indices]
            fold_val_df = full_train_df.iloc[val_indices]

            tokenized_train = construct_tokenized_dataset(
                fold_train_df, tokenizer, args.max_len, args.model_name
            )
            tokenized_val = construct_tokenized_dataset(
                fold_val_df, tokenizer, args.max_len, args.model_name
            )

            hate_train_dataset = hate_dataset(
                tokenized_train, fold_train_df["output"].values
            )
            hate_valid_dataset = hate_dataset(
                tokenized_val, fold_val_df["output"].values
            )

            # Fold별 경로 및 W&B 설정
            original_save_path = args.save_path
            original_run_name = args.run_name
            args.save_path = os.path.join(original_save_path, f"fold_{fold+1}")
            args.run_name = f"{original_run_name}_fold_{fold+1}"

            wandb.init(
                project="Hate_Speech_Detection", name=args.run_name, config=vars(args)
            )
            train(args, hate_train_dataset, hate_valid_dataset)  # 분할된 데이터셋 전달
            wandb.finish()

            args.save_path = original_save_path
            args.run_name = original_run_name

        print("=============== K-Fold Training Finished ================")

    else:
        # --- 단일 모델 학습 로직 ---
        print("--- Starting Single Model Training ---")
        wandb.init(
            project="Hate_Speech_Detection", name=args.run_name, config=vars(args)
        )
        train(args)  # model.py가 데이터를 직접 로드하도록 위임
        wandb.finish()
# .sh
# python main.py --run_name "ssac-bert" --lr 5e-4
# python main.py --run_name "ssac-bert" --lr 5e-3
# python main.py --model_name klue/roberta-large
