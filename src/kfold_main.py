# kfold_main.py (수정 완료)

import os
import argparse
import wandb
import numpy as np
from sklearn.model_selection import StratifiedKFold

from model import train, load_tokenizer_and_model_for_train
from data import load_data, construct_tokenized_dataset, hate_dataset
from arguments import add_common_args, add_train_args


def parse_args():
    """K-fold 학습을 위한 인자를 관리합니다."""
    parser = argparse.ArgumentParser(
        description="K-fold Cross-Validation Training Script"
    )
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of K-fold splits"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    print("--- Loading full training data for K-fold ---")
    full_train_df = load_data(args.dataset_name, "train", args.dataset_revision)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    labels = full_train_df["output"].values

    all_f1_scores = []

    print("--- Loading tokenizer ---")
    tokenizer, _ = load_tokenizer_and_model_for_train(args)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, labels)):
        fold_num = fold + 1
        print(f"\n=============== FOLD {fold_num}/{args.n_splits} ================")

        run_name_fold = f"{args.run_name}-fold{fold_num}"
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project="Hate_Speech_Detection_KFold",
            name=run_name_fold,
            config=vars(args),
            reinit=True,
        )

        train_df = full_train_df.iloc[train_idx]
        val_df = full_train_df.iloc[val_idx]
        train_labels = train_df["output"].values
        val_labels = val_df["output"].values

        tokenized_train = construct_tokenized_dataset(
            train_df, tokenizer, args.max_len, args.model_name
        )
        tokenized_val = construct_tokenized_dataset(
            val_df, tokenizer, args.max_len, args.model_name
        )

        hate_train_dataset = hate_dataset(tokenized_train, train_labels)
        hate_valid_dataset = hate_dataset(tokenized_val, val_labels)

        # [수정] args.save_dir -> args.save_path 로 모두 변경
        original_save_path = args.save_path
        args.save_path = os.path.join(original_save_path, f"fold_{fold_num}")

        best_f1 = train(args, hate_train_dataset, hate_valid_dataset)
        all_f1_scores.append(best_f1)

        args.save_path = original_save_path

    if wandb.run is not None:
        wandb.finish()
    print(f"\n=============== K-Fold Final Results ===============")
    print(f"F1 scores for each fold: {all_f1_scores}")
    print(f"Average F1 Score: {np.mean(all_f1_scores):.4f}")
    print(f"Standard Deviation: {np.std(all_f1_scores):.4f}")
