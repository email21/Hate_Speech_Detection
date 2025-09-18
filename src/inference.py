from torch.utils.data import DataLoader
import pandas as pd
import argparse
import torch
import os
import re
import numpy as np
from tqdm import tqdm
from model import load_model_for_inference
# from data import prepare_dataset
from data import construct_tokenized_dataset, hate_dataset 
from arguments import add_common_args, add_infer_args

def clean_data_for_inference(df):
    """
    Inference 시에 사용할 데이터 정제 함수
    data.py의 clean_data와 동일한 로직 적용
    """
    # 1. 유효한 태그 리스트 정의
    valid_tags = [
        '&location&', '&affiliation&', '&name&', '&company&', '&brand&', 
        '&art&', '&online-account&', '&address&', '&tel-num&', '&other&'
    ]
    
    # 2. '&'를 포함하는 모든 행을 찾음
    contains_ampersand = df['input'].str.contains('&', na=False)
    
    # 3. 유효한 태그 중 하나라도 포함하는 모든 행을 찾음
    valid_tags_pattern = '|'.join(re.escape(tag) for tag in valid_tags)
    contains_valid_tag = df['input'].str.contains(valid_tags_pattern, na=False)
    
    # 4. 삭제할 행 식별: '&'는 있지만, 유효한 태그는 없는 행
    rows_to_delete_mask = contains_ampersand & ~contains_valid_tag
    
    # 5. 해당 행들을 제외하고 데이터프레임을 새로 만듦
    cleaned_df = df[~rows_to_delete_mask].copy()
    
    print(f"원본 테스트 데이터 개수: {len(df)}")
    print(f"정제 후 테스트 데이터 개수: {len(cleaned_df)}")
    print(f"삭제된 데이터 개수: {len(df) - len(cleaned_df)}")
    
    return cleaned_df

def inference(model, tokenized_sent, device, batch_size):
    """
    학습된(trained) 모델을 통해 결과를 추론하는 function
    """
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
    return (np.concatenate(output_pred).tolist())

def infer_and_eval(args):
    """
    학습된 모델로 추론(infer)한 후에 예측한 결과(pred)를 평가(eval)
    """
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set model & tokenizer
    # tokenizer, model = load_model_for_inference(args.model_name, args.model_dir)
    # 저장된 경로(model_dir)만 사용하여 모델과 토크나이저를 불러옵니다.
    tokenizer, model = load_model_for_inference(args.model_dir)
    model.to(device)
    
    # 원본 테스트 데이터 로드
    original_test_df = pd.read_csv(os.path.join(args.dataset_dir, "test.csv"))
    
    # 예측을 위해 테스트 데이터를 정제
    cleaned_test_df = clean_data_for_inference(original_test_df)
    
    # 정제된 데이터를 토크나이징
    tokenized_test = construct_tokenized_dataset(cleaned_test_df, tokenizer, args.max_len, args.model_name)
    test_label = cleaned_test_df["output"].values # 라벨 사용되지는 않음
    hate_test_dataset = hate_dataset(tokenized_test, test_label)

    # 정제된 데이터로 답변 예측
    pred_answer = inference(
        model, hate_test_dataset, device, args.batch_size
    )
    print("--- Prediction done ---")
    
    # 예측 결과를 담은 데이터프레임 생성
    pred_df = pd.DataFrame({
        'id': cleaned_test_df['id'],
        'output': pred_answer
    })
    
     # 원본 테스트 데이터프레임과 예측 결과 데이터프레임을 'id' 기준으로 병합
    # how='left'를 사용하여 원본의 모든 행을 유지
    final_output = pd.merge(original_test_df[['id', 'input']], pred_df, on='id', how='left')

    # 예측값이 없는 행(삭제되었던 행)은 0으로 채움 (0: non-hate, 1: hate 가정)
    final_output['output'] = final_output['output'].fillna(0).astype(int)
    
    print(f"\n최종 제출 파일의 행 개수: {len(final_output)}")
    print("최종 제출 파일 샘플:")
    print(final_output.head())
    
    # 최종적으로 완성된 예측 라벨 csv 파일 형태로 저장
    result_path = args.result_dir
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    final_output.to_csv(os.path.join(result_path, "result.csv"), index=False)
    print("--- Save result ---")
    return final_output


    # # set data
    # _,_, hate_test_dataset, test_dataset = prepare_dataset(
    #     args.dataset_dir, tokenizer, args.max_len, args.model_name
    # )

    # predict answer
    # pred_answer = inference(model, hate_test_dataset, device, args.batch_size)   # model에서 class 추론
    # pred = pred_answer[0]
    # print("--- Prediction done ---")

    # make csv file with predicted answer
    # output = pd.DataFrame(
    #     {
    #         "id": test_dataset["id"],
    #         "input": test_dataset["input"],
    #         "output": pred,
    #     }
    # )

    # # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    # result_path = args.result_dir
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    # output.to_csv(os.path.join(result_path, "result.csv"), index=False)
    # print("--- Save result ---")
    # return output

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