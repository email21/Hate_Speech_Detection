# model.py

import pytorch_lightning as pl
import torch
from utils import compute_metrics
from data import prepare_dataset # data.py의 prepare_dataset은 HuggingFace Hub에서 데이터를 불러옵니다.
from datasets import Dataset # DataFrame을 Dataset으로 변환하기 위해 추가

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup


# ====================================================================
# 저장 오류 해결을 위한 Custom Trainer 클래스
# ====================================================================
class ContiguousTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        # 모델 저장 시 non-contiguous tensor 오류를 방지하기 위해 강제로 재정렬
        for name, param in self.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        super()._save(output_dir, state_dict)
# ====================================================================


def load_tokenizer_and_model_for_train(args):
    """(수정 없음) 학습을 위한 토크나이저와 모델 로딩"""
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    new_tokens = ["&name&", "&location&", "&affiliation&", "&company&", "&brand&", "&art&", "&other&", "&nama&", "&affifiation&", "&name", "&online-account&", "&compnay&", "&anme&", "& name&", "&address&", "&tel-num&", "&naem&"]
    tokenizer.add_tokens(new_tokens)
    
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    
    print("--- Tokenizer and Model for Train Loaded ---")
    return tokenizer, model


def load_model_for_inference(model_name, model_dir):
    """ 추론을 위한 모델 로딩"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_tokens = ["&name&", "&location&", "&affiliation&", "&company&", "&brand&", "&art&", "&other&", "&nama&", "&affifiation&", "&name", "&online-account&", "&compnay&", "&anme&", "& name&", "&address&", "&tel-num&", "&naem&"]
    tokenizer.add_tokens(new_tokens)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model


# train 함수가 args 외에 train_df, val_df를 직접 받도록 변경
def train(args, train_df, val_df):
    """모델을 학습(train)하고 best model을 저장"""
    pl.seed_everything(seed=args.seed, workers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(device)

    # 파일 로딩 대신, 전달받은 DataFrame을 HuggingFace Dataset으로 변환
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(val_df)

    def tokenize_dataset(dataset):
        tokenized_data = tokenizer(
            list(dataset["input"]),
            return_tensors="pt",
            max_length=args.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        tokenized_data["labels"] = list(dataset["output"])
        return tokenized_data

    hate_train_dataset = tokenize_dataset(train_dataset)
    hate_valid_dataset = tokenize_dataset(valid_dataset)

    # load_trainer_for_train 함수의 로직을 통합하고 ContiguousTrainer 사용
    training_args = TrainingArguments(
        output_dir=args.save_path, # main.py에서 fold별 경로를 지정
        save_total_limit=args.save_limit,
        save_strategy="epoch", # save_steps 대신 epoch 단위로 저장
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_strategy="epoch", # logging_steps 대신 epoch 단위로 로깅
        eval_strategy="epoch", # eval_steps 대신 epoch 단위로 평가
        load_best_model_at_end=True,
        metric_for_best_model="f1", # 평가 지표를 f1으로 설정
        greater_is_better=True,
        report_to="wandb",
        run_name=args.run_name,
        fp16=True,
    )

    MyCallback = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)
    
    trainer = ContiguousTrainer(
        model=model,
        args=training_args,
        train_dataset=hate_train_dataset,
        eval_dataset=hate_valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MyCallback],
    )
    
    print("--- Training Start ---")
    trainer.train()
    print("--- Training Finished ---")