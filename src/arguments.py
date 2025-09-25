import argparse

def add_common_args(parser):
    """
    학습과 추론에 공통으로 사용되는 인수를 추가하는 함수
    """
    parser.add_argument(
        "--dataset_name",
        type=str,
        #default="ensemble-2/NIKL_AU_2023_COMPETITION_v1.0",
        default="sagittarius5/NIKL_AU_2023_COMPETITION_v1.0",
        help="Hugging Face 데이터셋 이름 "
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=".././NIKL_AU_2023_COMPETITION_v1.0",
        help="데이터셋 디렉토리 경로",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="klue/bert-base",
        help="HuggingFace 모델 이름 (예: klue/bert-base)",
    )
    parser.add_argument(
        "--max_len", 
        type=int, 
        default=256, 
        help="입력 시퀀스의 최대 길이"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="배치 사이즈 (메모리에 맞게 조절 16,32)"
    )
    # parser.add_argument(
    #     "--model_dir",
    #     type=str,
    #     default=".././best_model",
    #     help="학습 시 모델을 저장하고, 추론 시 불러올 모델의 경로",
    # )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./best_model", # 경로 ./ 로 수정
        help="CPT 모델을 저장하고, 파인튜닝 시 불러올 모델의 경로",
    )
    return parser

def add_train_args(parser):
    """
    학습에만 사용되는 인수를 추가하는 함수
    """
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help='모델 타입 (예: "bert", "electra")',
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=".././model", 
        help="모델 저장 경로"
    )
    parser.add_argument(
        "--save_step", 
        type=int, 
        default=200, 
        help="모델을 저장할 스텝 간격"
    )
    parser.add_argument(
        "--logging_step", 
        type=int, 
        default=200, 
        help="로그를 출력할 스텝 간격"
    )
    parser.add_argument(
        "--eval_step", 
        type=int, 
        default=200, 
        help="모델을 평가할 스텝 간격"
    )
    parser.add_argument(
        "--save_limit", 
        type=int, 
        default=5, 
        help="저장할 모델의 최대 개수"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="랜덤 시드 값"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5, 
        help="에폭 수"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=3e-5, 
        help="학습률"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="가중치 감소(weight decay) 값"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=300, 
        help="워밍업 스텝 수"
    )
    parser.add_argument(
        "--scheduler", 
        type=str, 
        default="linear", 
        help="학습률 스케줄러 타입"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline-bert-test",
        help="wandb 에 기록되는 run name",
    )
    return parser

def add_infer_args(parser):
    """
    추론에만 사용되는 인수를 추가하는 함수
    """
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default=".././prediction", 
        help="결과 CSV 파일을 저장할 경로"
    )
    return parser