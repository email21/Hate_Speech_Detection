import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from utils import compute_metrics
from data import prepare_dataset
from model import load_tokenizer_and_model_for_train # 기존 함수 재사용

# 참고: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# --- Focal Loss를 사용하기 위한 Custom Trainer ---
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # 정답 레이블을 one-hot 인코딩으로 변환 (Focal Loss 입력 형식에 맞게)
        labels_one_hot = F.one_hot(labels, num_classes=self.model.config.num_labels).float()
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Focal Loss 인스턴스 생성 (alpha, gamma 값은 튜닝 대상)
        loss_fct = FocalLoss(alpha=0.25, gamma=2.0)
        loss = loss_fct(logits, labels_one_hot)
        
        return (loss, outputs) if return_outputs else loss

# --- train 함수 수정 ---
def train_with_custom_loss(args):
    """Focal Loss를 사용하여 모델을 학습하고 저장"""
    # fix a seed
    pl.seed_everything(seed=42, workers=False)

    # set device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # set model and tokenizer
    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(device)
   
    # HuggingFace 사용으로 prepare_dataset의 args.dataset_dir -> args.dataset_name
    hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
        prepare_dataset(args.dataset_name, tokenizer, args.max_len, args.model_name)
    )

    # set trainer (기존 load_trainer_for_train 대신 CustomTrainer를 직접 사용)
    training_args = TrainingArguments(
        output_dir=args.save_path + "/results",
        save_total_limit=args.save_limit,
        save_steps=args.save_step,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.save_path + "/logs",
        logging_steps=args.logging_step,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=args.run_name,
    )
    
    # Add callback & optimizer & scheduler
    my_callback = EarlyStoppingCallback(
        early_stopping_patience=2, early_stopping_threshold=0.001
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(hate_train_dataset) * args.epochs,
    )
    
    trainer = CustomTrainer( # <-
        model=model,
        args=training_args,
        train_dataset=hate_train_dataset,
        eval_dataset=hate_valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[my_callback],
        optimizers=(optimizer, scheduler),
    )

    print("--- Start train with Focal Loss ---")
    trainer.train()
    print("--- Finish train ---")
    model.save_pretrained(args.model_dir) # 모델 저장
    tokenizer.save_pretrained(args.model_dir) # 토크나이저 저장