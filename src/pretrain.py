"""
Phase 1: Continued Pre-training (CPT) for Domain Adaptation

RoBERTa 논문에서 나온 '더 많은 데이터'를 활용하여
기존 RoBERTa 모델을 대회 데이터셋 도메인에 맞게 추가로 사전 훈련합니다.
훈련된 모델은 이 대회만을 위한 '도메인 전문가'가 됩니다.
"""
import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from arguments import add_common_args
from data import prepare_dataset_for_cpt # CPT를 위한 데이터 로더 import

def parse_args():
    parser = argparse.ArgumentParser(description="Continued Pre-training arguments")
    parser = add_common_args(parser)
    parser.add_argument("--cpt_epochs", type=int, default=3, help="CPT 에포크 수")
    return parser.parse_args()

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    
    print("="*50)
    print("Phase 1: 도메인 적응을 위한 추가 사전 훈련 (CPT) 시작")
    print("="*50)

    # 1. 토크나이저 및 MLM 모델 로드
    print(f"기본 모델 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    # 2. CPT용 데이터셋 준비
    print("CPT용 데이터셋을 준비합니다...")
    lm_dataset = prepare_dataset_for_cpt(args.dataset_name, tokenizer, args.max_len)
    
    # 3. 데이터 콜레이터 설정 (Dynamic Masking)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # 4. CPT 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "cpt_checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.cpt_epochs,
        per_device_train_batch_size=args.batch_size // 2, # MLM은 메모리를 더 사용하므로 배치 사이즈 조정
        gradient_accumulation_steps=2,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
        fp16=True, # GPU 사용 시 학습 속도 향상을 위해 fp16 활성화
    )

    # 5. Trainer 초기화 및 훈련 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    print("CPT 훈련을 시작합니다...")
    trainer.train()
    
    # 6. 훈련된 모델과 토크나이저 저장
    print(f"CPT 완료! 도메인 적응 모델을 {args.model_dir} 경로에 저장합니다.")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print("Phase 1 완료")

if __name__ == "__main__":
    main()