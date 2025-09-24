import os
import argparse
import wandb

from model import train
# from tuned_model import train_with_custom_loss as train
from arguments import add_common_args, add_train_args


def parse_args():
    """
    학습(train)에 사용되는 arguments를 관리하는 함수
    """
    parser = argparse.ArgumentParser(description="Training arguments")
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    wandb.init(project="Hate_Speech_Detection", name=args.run_name)  # 혐오발언탐지
    train(args)

# .sh
# python main.py --run_name "ssac-bert" --lr 5e-4
# python main.py --run_name "ssac-bert" --lr 5e-3
# python main.py --model_name klue/roberta-large
