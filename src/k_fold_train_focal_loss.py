import os
import argparse
import wandb
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# model.py 대신 tuned_model.py에서 함수들 가져옴
from tuned_model import load_tokenizer_and_model_for_train, load_trainer_for_train
from data import load_data, construct_tokenized_dataset, hate_dataset
from utils import compute_metrics # compute_metrics를 명시적으로 사용하진 않지만, Trainer에 필요

def k_fold_train(args):
    """
    K-Fold 교차 검증을 사용하여 모델을 훈련하고 평가합니다. (Focal Loss 적용 버전)
    """
    # 1. 토크나이저 및 전체 데이터 로드
    tokenizer, _ = load_tokenizer_and_model_for_train(args)
    
    # K-Fold 분할을 위해 HuggingFace Hub에서 전체 학습 데이터셋을 로드합니다.
    full_train_df = load_data(args.dataset_name, "train", args.dataset_revision)
    if full_train_df is None:
        print("학습 데이터 로드에 실패했습니다. 스크립트를 종료합니다.")
        return

    # 2. StratifiedKFold 설정
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    fold_scores = []
    fold_losses = []

    # 3. K-Fold 루프 시작
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, full_train_df['output'])):
        print(f"===== Fold {fold+1}/{args.k_folds} 시작 =====")
        wandb.init(project="Hate_Speech_Detection_KFold", name=f"{args.run_name}-fold-{fold+1}", reinit=True)

        # 모델을 매 fold마다 새로 로드하여 가중치 초기화 보장
        _, model = load_tokenizer_and_model_for_train(args)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Fold별 학습 및 검증 데이터 분할
        train_df = full_train_df.iloc[train_idx]
        val_df = full_train_df.iloc[val_idx]

        # 데이터셋 생성
        tokenized_train = construct_tokenized_dataset(train_df, tokenizer, args.max_len, args.model_name)
        tokenized_val = construct_tokenized_dataset(val_df, tokenizer, args.max_len, args.model_name)
        
        hate_train_dataset = hate_dataset(tokenized_train, train_df['output'].values)
        hate_val_dataset = hate_dataset(tokenized_val, val_df['output'].values)
        
        # Focal Loss를 사용하는 Trainer를 로드
        trainer = load_trainer_for_train(args, model, hate_train_dataset, hate_val_dataset)
        # --------------------
        # 훈련 실행
        trainer.train()
        
        # 평가 실행
        eval_results = trainer.evaluate()
        
        # 결과 저장
        fold_f1_score = eval_results['eval_f1']
        fold_loss = eval_results['eval_loss']
        fold_scores.append(fold_f1_score)
        fold_losses.append(fold_loss)
        print(f"Fold {fold+1} F1 Score: {fold_f1_score:.4f}, Loss: {fold_loss:.4f}")
        
        wandb.log({f"fold_{fold+1}_eval_f1": fold_f1_score, f"fold_{fold+1}_eval_loss": fold_loss})
        wandb.finish()
        
        # 메모리 관리를 위해 모델과 트레이너 삭제
        del model
        del trainer
        torch.cuda.empty_cache()
        # fold_f1_score = eval_results['eval_f1']
        # fold_scores.append(fold_f1_score)
        # print(f"Fold {fold+1} F1 Score: {fold_f1_score}")
        
        # wandb.log({f"fold_{fold+1}_eval_f1": fold_f1_score})
        # wandb.finish()

    # 4. 최종 결과 출력
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    mean_loss = np.mean(fold_losses)
    print(f"\n===== K-Fold 최종 결과 (Focal Loss) =====*****")
    print(f"각 Fold의 F1 Score: {[round(s, 4) for s in fold_scores]}")
    print(f"평균 F1 Score: {mean_f1:.4f}")
    print(f"F1 Score 표준편차: {std_f1:.4f}")
    print(f"평균 Loss: {mean_loss:.4f}")
    
    # 최종 wandb에 평균 점수 기록
    wandb.init(project="Hate_Speech_Detection_KFold", name=f"{args.run_name}-summary", reinit=True)
    wandb.log({"mean_f1": mean_f1, "std_f1": std_f1})
    wandb.finish()

    # 5. 전체 데이터로 최종 모델 훈련
    print("\n========== 전체 데이터로 최종 모델 훈련 시작 ==========")
    wandb.init(project="Hate_Speech_Detection_KFold", name=f"{args.run_name}-final-train", reinit=True)
    
    # 새로운 최종 모델 초기화
    tokenizer, final_model = load_tokenizer_and_model_for_train(args)
    final_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 전체 학습 데이터를 다시 준비
    tokenized_train_full = construct_tokenized_dataset(full_train_df, tokenizer, args.max_len, args.model_name)
    hate_train_dataset_full = hate_dataset(tokenized_train_full, full_train_df['output'].values)
    
    # 최종 모델 Trainer는 eval_dataset 없이 설정
    final_trainer = load_trainer_for_train(args, final_model, hate_train_dataset_full, None)
    
    # 평가 없이 학습만 진행하도록 training_args 수정
    final_trainer.args.evaluation_strategy = "no"
    final_trainer.args.load_best_model_at_end = False

    final_trainer.train()
    
    print("최종 모델 훈련 완료.")
    final_model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print(f"최종 모델과 토크나이저가 '{args.model_dir}'에 저장되었습니다.")
    wandb.finish()
    
def parse_args():
    """
    이 스크립트 실행에 필요한 인자들을 파싱합니다.
    """
    parser = argparse.ArgumentParser(description="K-Fold Training with Focal Loss arguments")
    
    # 다른 파일에 정의된 공통 인자들을 가져와서 추가합니다.
    from arguments import add_common_args, add_train_args
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    
    # 이 스크립트에서만 사용하는 K-Fold를 위한 인수를 특별히 추가합니다.
    parser.add_argument(
        "--k_folds", 
        type=int, 
        default=5, 
        help="K-Fold 교차 검증의 폴드 수 (K값)"
    )
    
    args = parser.parse_args()
    return args
    
    # print(f"\n===== K-Fold 최종 결과 =====")
    # print(f"각 Fold의 F1 Score: {fold_scores}")
    # print(f"평균 F1 Score: {mean_f1:.4f}")
    # print(f"F1 Score 표준편차: {std_f1:.4f}")

    # #  전체 데이터로 최종 모델 재학습
    # print("\n===== 전체 데이터로 최종 모델 재학습 시작 =====")
    # # 모델과 토크나이저를 다시 로드하여 깨끗한 상태에서 시작
    # tokenizer, final_model = load_tokenizer_and_model_for_train(args)
    # final_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # # 전체 학습 데이터를 다시 준비
    # tokenized_train_full = construct_tokenized_dataset(full_train_df, tokenizer, args.max_len, args.model_name)
    # hate_train_dataset_full = hate_dataset(tokenized_train_full, full_train_df['output'].values)
    
    # # 최종 모델 Trainer는 eval_dataset 없이 설정 (Focal Loss 버전)
    # final_trainer = load_trainer_for_train_with_focal_loss(args, final_model, hate_train_dataset_full, None)
    
    # # 평가 없이 학습만 진행하도록 training_args 수정
    # final_trainer.args.evaluation_strategy = "no"
    # final_trainer.args.load_best_model_at_end = False

    # final_trainer.train()
    
    # print("최종 모델 훈련 완료.")
    # final_model.save_pretrained(args.model_dir)
    # tokenizer.save_pretrained(args.model_dir)
    # print(f"최종 모델과 토크나이저가 '{args.model_dir}'에 저장되었습니다.")
    
# def main(args):
#     k_fold_train(args)

# if __name__ == '__main__':
#     # 여기에 argument 파싱 코드를 넣어주세요. (e.g., from arguments import...)
#     # 예시:
#     from arguments import add_common_args, add_train_args
#     parser = argparse.ArgumentParser()
#     add_common_args(parser)
#     add_train_args(parser)
#     args = parser.parse_args()
    
#     # wandb 로그인
#     # wandb.login(key=args.wandb_key)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    k_fold_train(args)