import pytorch_lightning as pl
import torch
from utils import compute_metrics
from data import prepare_dataset

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

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    print(model_config)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    print("--- Modeling Done ---")
    return tokenizer, model

def load_model_for_inference(model_dir):
    """
    추론(infer)에 필요한 모델과 토크나이저를 저장된 경로(model_dir)에서 load
    불필요한 model_name 매개변수를 제거하여 코드를 명확화
    """
    # 저장된 모델과 토크나이저를 같은 디렉토리에서 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    return tokenizer, model

# def load_model_for_inference(model_name,model_dir):
#     """추론(infer)에 필요한 모델과 토크나이저 load """
#     # load tokenizer
#     Tokenizer_NAME = model_name
#     # tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME) # 원본 토크나이저저
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)  # 훈련 후 저장된 모델 디렉토리에서 토크나이저를 로드

#     ## load my model
#     model = AutoModelForSequenceClassification.from_pretrained(model_dir)

#     return tokenizer, model

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
        eval_strategy="steps",  # evaluation strategy to adopt during training
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

    trainer = Trainer(
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

    # set model and tokenizer
    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(device)
    
    special_tokens = ["&location&", "&affiliation&", "&name&", "&company&", "&brand&", 
                      "&art&", "&online-account&", "&address&", "&tel-num&", "&other&"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # set data
    # hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
        # prepare_dataset(args.dataset_dir, tokenizer, args.max_len)
       # hate_test_dataset과 test_dataset 변수가 선언만 되고 실제로는 사용되지 않고 있음
    # train 함수는 학습과 검증 데이터셋만 필요하므로, 나머지는 _로 받기
    hate_train_dataset, hate_valid_dataset, _, _ = prepare_dataset(
        args.dataset_dir, tokenizer, args.max_len, args.model_name
    )  
# --- data loading Done ---data tokenizing Done ---pytorch dataset class Done ---

    # set trainer
    trainer = load_trainer_for_train(
        args, model, hate_train_dataset, hate_valid_dataset
    )

    # train model
    print("--- Start train ---")
    trainer.train()
    print("--- Finish train ---")
    model.save_pretrained(args.model_dir) # 모델 저장
    tokenizer.save_pretrained(args.model_dir) # 토크나이저 저장