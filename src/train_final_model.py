import os
import argparse
import wandb
import torch
import pytorch_lightning as pl

# Focal Loss가 적용된 모델/트레이너와 데이터 로더를 가져옵니다.
from tuned_model import load_tokenizer_and_model_for_train, load_trainer_for_train
from data import load_data, construct_tokenized_dataset, hate_dataset

def train_final_model(args):
    """
    전체 학습 데이터를 사용하여 최종 모델을 학습하고 저장합니다.
    """
    # 1. Wandb 초기화
    wandb.init(project="Hate_Speech_Detection_KFold", name=f"{args.run_name}-fold-{fold+1}", reinit=True)
    print(f"--- Wandb Run: {args.run_name}_final_training ---")

    # 2. 시드 고정
    pl.seed_everything(seed=args.seed, workers=False)

    # 3. 토크나이저 및 모델 로드
    print("--- Loading Tokenizer & Model ---")
    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 4. 전체 학습 데이터 로드 (K-Fold 때와 동일)
    print("--- Loading Full Training Data ---")
    full_train_df = load_data(args.dataset_name, "train", args.dataset_revision)
    if full_train_df is None:
        print("학습 데이터 로드에 실패했습니다. 스크립트를 종료합니다.")
        return

    # 5. 데이터셋 생성
    print("--- Creating Dataset for Final Training ---")
    tokenized_train_full = construct_tokenized_dataset(full_train_df, tokenizer, args.max_len, args.model_name)
    hate_train_dataset_full = hate_dataset(tokenized_train_full, full_train_df['output'].values)

    # 6. 최종 학습을 위한 Trainer 설정
    # 검증(validation) 데이터 없이 오직 학습 데이터만 사용
    print("--- Setting up Trainer for Final Training ---")
    final_trainer = load_trainer_for_train(args, model, hate_train_dataset_full, None)
    
    # 7. Trainer 옵션 수정: 평가 없이 오직 학습에만 집중
    final_trainer.args.evaluation_strategy = "no"
    final_trainer.args.load_best_model_at_end = False
    final_trainer.args.save_strategy = "no" # 중간 저장은 필요 없습니다.

    # 8. 최종 학습 시작
    print("--- Starting Final Training ---")
    final_trainer.train()
    print("--- Final Training Finished ---")

    # 9. 최종 모델과 토크나이저 저장
    print(f"--- Saving Final Model & Tokenizer to '{args.model_dir}' ---")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    final_trainer.save_model(args.model_dir) # trainer.save_model() 사용 권장
    tokenizer.save_pretrained(args.model_dir)
    
    # model.save_pretrained(args.model_dir)
    # tokenizer.save_pretrained(args.model_dir) # 토크나이저 저장
    
    print("--- All Done ---")
    wandb.finish()

def parse_args():
    """
    이 스크립트 실행에 필요한 인자들을 파싱합니다.
    """
    parser = argparse.ArgumentParser(description="Final Model Training arguments")
    from arguments import add_common_args, add_train_args
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # if args.wandb_key:
    #     wandb.login(key=args.wandb_key)
    
    train_final_model(args)


#python train_final_model.py --model_dir ./saved_models/roberta-final-with-focal-loss 
#--run_name "roberta-final-focal-loss" --dataset_name "ensemble-2/AEDA-dataset" --dataset_revision "v1.1"

#    --epochs 10 \
#     --lr 2e-5 \
#     --weight_decay 0.1

#--dataset_revision "aeda"
