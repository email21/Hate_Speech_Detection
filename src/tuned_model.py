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
        # inputs: 모델의 로짓 (batch_size, num_classes)
        # targets: 실제 레이블 (batch_size)
        
        # 1. CrossEntropyLoss를 계산합니다. (내부적으로 softmax 포함)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. 정답 클래스에 대한 예측 확률(pt)을 계산합니다.
        pt = torch.exp(-ce_loss)
        
        # 3. Focal Loss를 계산합니다.
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        # 4. 지정된 reduction에 따라 최종 손실 값을 반환합니다.
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    # def forward(self, inputs, targets):
    #     # Focal Loss는 클래스가 2개일 때 binary_cross_entropy를 사용합니다.
    #     # 만약 다중 클래스 분류라면 cross_entropy를 사용해야 합니다.
    #     BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
    #     pt = torch.exp(-BCE_loss)
    #     F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

    #     if self.reduction == 'mean':
    #         return torch.mean(F_loss)
    #     elif self.reduction == 'sum':
    #         return torch.sum(F_loss)
    #     else:
    #         return F_loss

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss() # FocalLoss 인스턴스 생성
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 기본 CrossEntropyLoss 대신 FocalLoss를 사용하여 손실을 계산합니다.
        # 우리의 문제는 이진 분류(혐오/비혐오)이므로 logits과 labels를 그대로 사용합니다.
        # 레이블이 0 또는 1로 구성되어 있다고 가정합니다.
        loss = self.focal_loss(logits.squeeze(-1), labels) 
        
        return (loss, outputs) if return_outputs else loss

# ==================== 이 함수가 추가되었습니다! ====================
def load_trainer_for_train(args, model, hate_train_dataset, hate_valid_dataset):
    """
    Focal Loss를 사용하는 CustomTrainer를 생성하고 반환합니다.
    """
    print("--- Set TrainingArguments ---")
    training_args = TrainingArguments(
        output_dir=args.save_path,
        overwrite_output_dir=True,
        save_total_limit=args.save_limit,
        save_strategy="steps",
        save_steps=args.save_step,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.save_path + "/logs",
        logging_steps=args.logging_step,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=args.run_name,
    )
    
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
    
    # hate_train_dataset이 None일 경우(최종 학습) 에러 방지
    num_training_steps = (
        len(hate_train_dataset) * args.epochs
        if hate_train_dataset is not None
        else args.epochs
    )
    
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # --- 핵심 변경점 ---
    # 기본 Trainer 대신 Focal Loss가 적용된 CustomTrainer를 사용합니다.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=hate_train_dataset,
        eval_dataset=hate_valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[my_callback],
        optimizers=(optimizer, scheduler),
    )
    # --------------------
    
    print("--- Set CustomTrainer with Focal Loss Done ---")
    return trainer