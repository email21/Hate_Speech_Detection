import os
import argparse
import pandas as pd
import wandb

from model import train

# from tuned_model import train_with_custom_loss as train
from arguments import add_common_args, add_train_args


def parse_args():
    """
    학습(train)에 사용되는 arguments를 관리하는 함수
    """
    # parser = argparse.ArgumentParser(description="Training arguments")
    # parser = add_common_args(parser)
    # parser = add_train_args(parser)
    # args = parser.parse_args()
    # return args

    parser = argparse.ArgumentParser(description="Training arguments")
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    # K-Fold를 위한 인자 추가
    parser.add_argument(
        "--n_splits",
        type=int,
        default=0,
        help="K-Fold 분할 수. 0 또는 1이면 K-Fold를 수행하지 않음.",
    )
    return parser.parse_args()


# if __name__ == "__main__":
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     args = parse_args()
#     wandb.init(project="Hate_Speech_Detection", name=args.run_name)  # 혐오발언탐지
#     train(args)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    if args.n_splits > 1:
        # --- K-Fold 로직 시작 ---
        print(f"--- Starting {args.n_splits}-Fold Cross Validation ---")
        train_df = pd.read_csv(os.path.join(args.dataset_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(args.dataset_dir, "dev.csv"))
        combined_df = pd.concat([train_df, dev_df]).reset_index(drop=True)

        kf = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=args.seed
        )

        os.makedirs(args.save_path, exist_ok=True)

        for fold, (train_indices, val_indices) in enumerate(
            kf.split(combined_df, combined_df["output"])
        ):
            print(f"=============== FOLD {fold+1}/{kf.get_n_splits()} ================")
            fold_train_df = combined_df.iloc[train_indices]
            fold_val_df = combined_df.iloc[val_indices]

            original_save_path = args.save_path
            original_run_name = args.run_name

            args.save_path = os.path.join(original_save_path, f"fold_{fold+1}")
            args.run_name = f"{original_run_name}_fold_{fold+1}"

            wandb.init(
                project="Hate_Speech_Detection", name=args.run_name, config=vars(args)
            )
            train(args, fold_train_df, fold_val_df)
            wandb.finish()

            args.save_path = original_save_path
            args.run_name = original_run_name
        print("=============== K-Fold Training Finished ================")

    else:
        # --- 단일 모델 학습 로직 ---
        print("--- Starting Single Model Training ---")
        train_df = pd.read_csv(os.path.join(args.dataset_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(args.dataset_dir, "dev.csv"))

        wandb.init(
            project="Hate_Speech_Detection", name=args.run_name, config=vars(args)
        )
        train(args, train_df, dev_df)
        wandb.finish()


# .sh
# python main.py --run_name "ssac-bert" --lr 5e-4
# python main.py --run_name "ssac-bert" --lr 5e-3
# python main.py --model_name klue/roberta-large
