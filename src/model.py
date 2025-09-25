import pytorch_lightning as pl
import torch
from utils import compute_metrics
from data import prepare_dataset, prepare_kfold_dataset, construct_tokenized_dataset, hate_dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup


def load_tokenizer_and_model_for_train(args):
    """학습(train)을 위한 사전학습(pretrained) 토크나이저와 모델을 huggingface에서 load"""
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 비식별화된 단어 토큰으로 추가
    new_tokens = ['&name&', '&location&', '&affiliation&', '&company&', '&brand&', '&art&', '&other&', '&nama&', '&affifiation&', '&name', '&online-account&', '&compnay&', '&anme&', '& name&', '&address&', '&tel-num&', '&naem&']
    tokenizer.add_tokens(new_tokens)
    
    print("토큰 추가 확인:", tokenizer.convert_tokens_to_ids(new_tokens)) # 토큰이 잘 추가되었는지 확인 # [32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010, 32011, 32012, 32013, 32014, 32015, 32016]
    print("늘어난 토큰 크기:", len(tokenizer)) # 토큰 크기 늘어난 것 확인 # 원래는 32000개

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    print(model_config)
    
    # # dropout 추가
    # model_config = AutoConfig.from_pretrained(MODEL_NAME)
    # model_config.num_labels = 2
    # model_config.hidden_dropout_prob = args.dropout_rate  # 드롭아웃 비율 설정
    # model_config.attention_probs_dropout_prob = args.dropout_rate

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    
    # 모델의 임베딩 레이어 크기 조정(토큰을 늘렸기 때문에)
    model.resize_token_embeddings(len(tokenizer))
    print("임베딩 레이어 크기 조정 완료")
    
    print("--- Modeling Done ---")
    return tokenizer, model

# K-Fold를 위한 함수(load_tokenizer_and_model_for_train 대신)
def load_tokenizer_and_model_for_kfold(model_name):
    """
    K-Fold의 각 fold마다 새로운 모델과 토크나이저를 로드하는 함수
    (기존 load_tokenizer_and_model_for_train 함수와 거의 동일)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 비식별화 토큰 추가 (기존 코드와 동일)
    new_tokens = ['&name&', '&location&', '&affiliation&', '&company&', '&brand&', '&art&', '&other&', '&nama&', '&affifiation&', '&name', '&online-account&', '&compnay&', '&anme&', '& name&', '&address&', '&tel-num&', '&naem&']
    tokenizer.add_tokens(new_tokens)

    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model

def load_model_for_inference(model_name,model_dir):
    """추론(infer)에 필요한 모델과 토크나이저 load """
    # load tokenizer
    Tokenizer_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # 비식별화된 단어 토큰으로 추가
    new_tokens = ['&name&', '&location&', '&affiliation&', '&company&', '&brand&', '&art&', '&other&', '&nama&', '&affifiation&', '&name', '&online-account&', '&compnay&', '&anme&', '& name&', '&address&', '&tel-num&', '&naem&']
    tokenizer.add_tokens(new_tokens)
    
    print("토큰 추가 확인2:", tokenizer.convert_tokens_to_ids(new_tokens)) # 토큰이 잘 추가되었는지 확인
    print("늘어난 토큰 크기2:", len(tokenizer)) # 토큰 크기 늘어난 것 확인
    
    ## load my model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    model.resize_token_embeddings(len(tokenizer))
    print("추론용 임베딩 레이어 크기 조정 완료")

    return tokenizer, model

# beomi-KcELECTRA-base-v2022 실행할 때 오류나서 추가함
class ContiguousTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        # contiguous 추가
        for name, param in self.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        super()._save(output_dir, state_dict)


def load_trainer_for_train(args, model, hate_train_dataset, hate_valid_dataset):
    """학습(train)을 위한 huggingface trainer 설정"""
    training_args = TrainingArguments(
        output_dir=args.save_path + "/results",  # output directory
        save_total_limit=args.save_limit,  # number of total save model.
        save_steps=args.save_step,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.save_path + "logs",  # directory for storing logs
        logging_steps=args.logging_step,  # log saving step.
        eval_strategy="steps",  # eval strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=args.eval_step,  # evaluation step.
        load_best_model_at_end=True,
        report_to="wandb",  # W&B 로깅 활성화
        run_name=args.run_name,  # run_name 지정
    )

    ## Add callback & optimizer & scheduler
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    print("--- Set training arguments Done ---")

    trainer = ContiguousTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=hate_train_dataset,  # training dataset
        eval_dataset=hate_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=len(hate_train_dataset) * args.epochs,
            ),
        ),
    )
    print("--- Set Trainer Done ---")

    return trainer


def train(args):
    """모델을 학습(train)하고 best model을 저장"""
    # fix a seed
    pl.seed_everything(seed=42, workers=False)

    # set device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.n_splits > 1:
        print(f"--- Starting {args.n_splits}-Fold Cross-Validation ---")
        
        # 1. K-Fold를 위한 전체 데이터 로드
        DATASET_REVISION = args.revision
        full_train_df, _ = prepare_kfold_dataset(args.dataset_name, DATASET_REVISION)

        # 2. StratifiedKFold 설정
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        labels = full_train_df["output"].values
        
        all_fold_metrics = []

        # 3. K-Fold 루프 시작
        for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, labels)):
            print(f"\n========== Fold {fold + 1}/{args.n_splits} ==========")

            # 4. Fold마다 모델과 토크나이저를 새로 로드
            tokenizer, model = load_tokenizer_and_model_for_kfold(args.model_name)
            model.to(device)

            # 5. 현재 Fold에 해당하는 데이터 분할 및 Pytorch Dataset 생성
            train_df = full_train_df.iloc[train_idx]
            valid_df = full_train_df.iloc[val_idx]
            
            tokenized_train = construct_tokenized_dataset(train_df, tokenizer, args.max_len, args.model_name)
            tokenized_valid = construct_tokenized_dataset(valid_df, tokenizer, args.max_len, args.model_name)

            hate_train_dataset = hate_dataset(tokenized_train, train_df["output"].values)
            hate_valid_dataset = hate_dataset(tokenized_valid, valid_df["output"].values)
            
            # 6. Fold별 TrainingArguments 설정
            fold_output_dir = os.path.join(args.save_path, f"fold_{fold+1}")
            fold_run_name = f"{args.run_name}_fold_{fold+1}"

            training_args = TrainingArguments(
                output_dir=fold_output_dir,
                run_name=fold_run_name,
                # ... (나머지 인자들은 이전과 동일하게 설정) ...
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                logging_dir=os.path.join(fold_output_dir, "logs"),
                logging_steps=args.logging_step,
                eval_strategy="steps",
                eval_steps=args.eval_step,
                save_steps=args.save_step,
                save_total_limit=args.save_limit,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                report_to="wandb",
            )
            
            # Trainer 설정 (ContiguousTrainer 사용)
            trainer = ContiguousTrainer(
                model=model,
                args=training_args,
                train_dataset=hate_train_dataset,
                eval_dataset=hate_valid_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            # 7. 현재 Fold 학습 시작
            trainer.train()

            # 8. Fold 평가 및 결과 저장
            metrics = trainer.evaluate(eval_dataset=hate_valid_dataset)
            all_fold_metrics.append(metrics)
            
            # 9. Fold별 베스트 모델 저장
            best_model_path = os.path.join(args.model_dir, f"best_model_fold_{fold+1}")
            trainer.save_model(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"Best model for fold {fold+1} saved to {best_model_path}")

        # 10. 최종 평균 점수 출력
        print("\n========== K-Fold Cross-Validation Summary ==========")
        avg_eval_f1 = np.mean([m["eval_f1"] for m in all_fold_metrics])
        print(f"Average F1 Score across {args.n_splits} folds: {avg_eval_f1:.4f}")

    # ==================================================================
    # ## 일반 학습 로직 (n_splits <= 1)
    # ==================================================================
    else:
        print("--- Starting Single Training Run (K-Fold is not used) ---")

        # 1. 모델/토크나이저 로드
        tokenizer, model = load_tokenizer_and_model_for_train(args)
        model.to(device)

        # 2. 데이터셋 준비
        hate_train_dataset, hate_valid_dataset, _, _ = prepare_dataset(
            args.dataset_name, tokenizer, args.max_len, args.model_name, args.revision
        )

        # 3. Trainer 준비
        trainer = load_trainer_for_train(
            args, model, hate_train_dataset, hate_valid_dataset
        )

        # 4. 학습 시작
        trainer.train()
        print("--- Finish train ---")
        
        # 5. 최종 모델 저장
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)