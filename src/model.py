# model.py

import pytorch_lightning as pl
import torch
from utils import compute_metrics
from data import prepare_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


class ContiguousTrainer(Trainer):
    """KcELECTRA 계열 모델의 저장 오류를 해결하기 위한 Custom Trainer"""

    def _save(self, output_dir=None, state_dict=None):
        for name, param in self.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        super()._save(output_dir, state_dict)


def load_tokenizer_and_model_for_train(args):
    """학습을 위한 토크나이저와 모델 로딩 (수정 없음)"""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.model_revision
    )
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

    model_config = AutoConfig.from_pretrained(
        args.model_name, revision=args.model_revision
    )
    model_config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=model_config, revision=args.model_revision
    )
    model.resize_token_embeddings(len(tokenizer))

    print("--- Tokenizer and Model for Train Loaded ---")
    return tokenizer, model


# [핵심 수정] K-Fold와 단일 학습 모두를 처리할 수 있도록 hate_train_dataset=None을 기본값으로 설정
def train(args, hate_train_dataset=None, hate_valid_dataset=None):
    """모델을 학습하고 best model을 저장"""
    pl.seed_everything(seed=args.seed, workers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(device)

    # 데이터셋이 인자로 주어지지 않은 경우(단일 학습)에만 data.py를 통해 직접 로드
    if hate_train_dataset is None or hate_valid_dataset is None:
        print("--- Loading data for a single run ---")
        hate_train_dataset, hate_valid_dataset, _, _ = prepare_dataset(
            args.dataset_name,
            tokenizer,
            args.max_len,
            args.model_name,
            revision=args.dataset_revision,
        )

    training_args = TrainingArguments(
        output_dir=args.save_path + "/results",
        save_total_limit=args.save_limit,
        save_strategy="steps",
        save_steps=args.save_step,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.save_path + "/logs",
        logging_strategy="steps",
        logging_steps=args.logging_step,
        evaluation_strategy="steps",
        eval_steps=args.eval_step,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        run_name=args.run_name,
        fp16=True,
    )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=10, early_stopping_threshold=0.001
    )

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
