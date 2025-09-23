# main.py

import os
import argparse
import wandb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model import train
from arguments import add_common_args, add_train_args

def parse_args():
    """학습에 사용되는 arguments를 관리"""
    parser = argparse.ArgumentParser(description="Training arguments")
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    return parser.parse_args()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    # --- K-Fold 로직 시작 ---
    # 로컬 CSV 파일(train, dev)을 합쳐서 K-Fold를 수행
    train_df = pd.read_csv(os.path.join(args.dataset_dir, "train.csv"))
    dev_df = pd.read_csv(os.path.join(args.dataset_dir, "dev.csv"))
    combined_df = pd.concat([train_df, dev_df]).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_indices, val_indices) in enumerate(kf.split(combined_df, combined_df["output"])):
        print(f"=============== FOLD {fold+1}/{kf.get_n_splits()} ================")

        fold_train_df = combined_df.iloc[train_indices]
        fold_val_df = combined_df.iloc[val_indices]

        # 각 Fold별로 모델 저장 경로와 wandb 실행 이름을 다르게 설정
        original_save_path = args.save_path
        original_run_name = args.run_name
        
        args.save_path = os.path.join(original_save_path, f"fold_{fold+1}")
        args.run_name = f"{original_run_name}_fold_{fold+1}"

        # W&B 초기화 (각 Fold를 별개의 Run으로 기록)
        wandb.init(project="Hate_Speech_Detection", name=args.run_name, config=vars(args))

        # 수정된 train 함수 호출 (분할된 데이터프레임을 직접 전달)
        train(args, fold_train_df, fold_val_df)

        # 다음 Fold를 위해 원래 경로와 이름으로 복원
        args.save_path = original_save_path
        args.run_name = original_run_name
        wandb.finish() # 현재 Fold의 W&B run 종료

    print("=============== K-Fold Training Finished ================")