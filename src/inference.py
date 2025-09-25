from torch.utils.data import DataLoader
import pandas as pd
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from model import load_model_for_inference
from data import prepare_dataset
from arguments import add_common_args, add_infer_args


def inference(model, tokenized_sent, device, batch_size=32):
    """학습된(trained) 모델을 통해 결과를 추론하는 function"""
    dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
    model.eval()
    output_pred = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
    return (np.concatenate(output_pred).tolist(),)


def infer_and_eval(args):
    """학습된 모델로 추론(infer)한 후에 예측한 결과(pred)를 평가(eval)"""
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set model & tokenizer
    tokenizer, model = load_model_for_inference(
        args.model_name, args.model_dir, args.model_revision
    )
    # tokenizer, model = load_model_for_inference(model_name,model_dir)
    model.to(device)

    # set data
    # _,_, hate_test_dataset, test_dataset = prepare_dataset("./NIKL_AU_2023_COMPETITION_v1.0",tokenizer,256)
    # Hugging Face Hub에서 test 데이터셋 로드
    print(f"--- Loading test dataset from Hugging Face Hub: {args.dataset_name} ---")
    # test_dataset = load_data(args.dataset_name, split="test")
    # HuggingFace 사용으로 prepare_dataset의 args.dataset_dir -> args.dataset_name
    _, _, hate_test_dataset, test_dataset = prepare_dataset(
        args.dataset_name,
        tokenizer,
        args.max_len,
        args.model_name,
        revision=args.dataset_revision,  # 이 부분 추가
    )

    # predict answer
    pred_answer = inference(model, hate_test_dataset, device)  # model에서 class 추론
    pred = pred_answer[0]
    print("--- Prediction done ---")

    # make csv file with predicted answer
    output = pd.DataFrame(
        {
            "id": test_dataset["id"],
            "input": test_dataset["input"],
            "output": pred,
        }
    )

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    result_path = "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "result.csv"), index=False)
    print("--- Save result ---")
    return output


def parse_args():
    """
    추론(inference)에 사용되는 arguments를 관리하는 함수
    """
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser = add_common_args(parser)
    parser = add_infer_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    infer_and_eval(args)


# if __name__ == "__main__":
#     model_name = "klue/bert-base"
#     model_dir = "./best_model"

#     infer_and_eval(model_name,model_dir)
