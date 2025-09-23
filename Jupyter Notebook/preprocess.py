# preprocess.py (slang_dic 사전 제거, PyTorch 교정 모델 최종 버전)

import pandas as pd
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pandas의 progress_apply를 사용하기 위한 설정
tqdm.pandas()


def preprocess(text):
    """PyTorch 맞춤법/띄어쓰기 교정 및 정규식을 사용한 전처리 함수."""
    if not isinstance(text, str):
        return ""

    # 1단계: 단독 사용 'ㅗ' 처리 및 불필요한 특수 문자 제거
    # text = re.sub(r"(^|\s)[ㅗ]($|\s)", " 모욕 ", text)

    # 2단계: 한글, 영어, 숫자, 그리고 지정된 특수문자들(.,!?&)`을 제외한 모든 문자를 공백으로 변경
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9.,!?&\s]", " ", text)

    # 3단계: 최종적으로 여러 개의 공백을 하나로 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    """CSV 파일만 전처리하고 기존 파일을 덮어씁니다."""
    dataset_dir = "../NIKL_AU_2023_COMPETITION_v1.0"
    filenames = ["train.csv", "dev.csv", "test.csv"]

    for filename in filenames:
        file_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(file_path):
            print(f"경고: {file_path}를 찾을 수 없습니다. 건너뜁니다.")
            continue

        print(f"--- CSV 전처리 시작: {file_path} ---")
        df = pd.read_csv(file_path)
        df["input"] = df["input"].progress_apply(preprocess)
        df.dropna(subset=["input"], inplace=True)
        df = df[df["input"].str.len() > 0]
        df.to_csv(file_path, index=False, encoding="utf-8")
        print(f"--- 전처리 완료 및 저장: {file_path} ---")


if __name__ == "__main__":
    main()
