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

from transformers import TrainerCallback


def load_tokenizer_and_model_for_train(args):
    """학습(train)을 위한 사전학습(pretrained) 토크나이저와 모델을 huggingface에서 load"""
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 비식별화된 단어 토큰으로 추가
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

    print(
        "토큰 추가 확인:", tokenizer.convert_tokens_to_ids(new_tokens)
    )  # 토큰이 잘 추가되었는지 확인 # [32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010, 32011, 32012, 32013, 32014, 32015, 32016]
    print(
        "늘어난 토큰 크기:", len(tokenizer)
    )  # 토큰 크기 늘어난 것 확인 # 원래는 32000개

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    print(model_config)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )

    # 모델의 임베딩 레이어 크기 조정(토큰을 늘렸기 때문에)
    model.resize_token_embeddings(len(tokenizer))
    print("임베딩 레이어 크기 조정 완료")

    print("--- Modeling Done ---")
    return tokenizer, model


def load_model_for_inference(model_name, model_dir):
    """추론(infer)에 필요한 모델과 토크나이저 load"""
    # load tokenizer
    Tokenizer_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # 비식별화된 단어 토큰으로 추가
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

    print(
        "토큰 추가 확인2:", tokenizer.convert_tokens_to_ids(new_tokens)
    )  # 토큰이 잘 추가되었는지 확인
    print("늘어난 토큰 크기2:", len(tokenizer))  # 토큰 크기 늘어난 것 확인

    ## load my model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # 모델의 임베딩 레이어 크기 조정(토큰을 늘렸기 때문에)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


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
    print("--- Set training arguments Done ---")

    # trainer = Trainer(
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=hate_train_dataset,  # training dataset
        eval_dataset=hate_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[
            MyCallback,
        ],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=len(hate_train_dataset) * args.epochs,
            ),
        ),
    )

    # 트레이너의 _save 메소드를 직접 교체하여 오류를 원천 차단합니다.
    # ==============================================================================
    original_save = trainer._save

    def new_save(output_dir=None, state_dict=None):
        # 1. 저장 직전에 모든 파라미터의 메모리를 강제로 연속적으로 만듭니다.
        for name, param in trainer.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        # 2. 메모리 재정렬 후, 원래의 저장 로직을 호출합니다.
        original_save(output_dir, state_dict)

    trainer._save = new_save
    # ===========================================================================

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

    # set data
    # hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
    #     prepare_dataset(args.dataset_dir, tokenizer, args.max_len)
    # )

    # HuggingFace 사용으로 prepare_dataset의 args.dataset_dir -> args.dataset_name
    hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
        prepare_dataset(args.dataset_name, tokenizer, args.max_len, args.model_name)
    )

    # set trainer
    trainer = load_trainer_for_train(
        args, model, hate_train_dataset, hate_valid_dataset
    )

    # train model
    print("--- Start train ---")
    trainer.train()
    print("--- Finish train ---")
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)  # 토크나이저 저장
