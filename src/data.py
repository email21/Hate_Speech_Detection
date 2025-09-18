import os
import pandas as pd
import torch
import re

class hate_dataset(torch.utils.data.Dataset):
    """dataframe을 torch dataset class로 변환"""

    def __init__(self, hate_dataset, labels):
        self.dataset = hate_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(dataset_dir):
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)
    print("dataframe 의 형태")
    print("-" * 100)
    print(dataset.head())
    return dataset

def clean_data(df): # 추가
    """데이터 정제: 오타 수정 및 불필요한 행 제거"""
    # 휴먼 에러 토큰 오타 정정
    typo_map = {
        '&anme&': '&name&',
        '&naem&': '&name&',
        '& name&': '&name&',
        '&nama&': '&name&',
        '&affifiation&': '&affiliation&',
        '&affilation&': '&affiliation&',
        '&compnay&': '&company&'
    }
    
    # apply(lambda x: ...)를 사용해 모든 행에 대해 오타 정정 적용
    for typo, correct in typo_map.items():
        if 'input' in df.columns:
            df['input'] = df['input'].apply(lambda x: str(x).replace(typo, correct) if pd.notna(x) else x)
    
    print("\n'input' 열의 휴먼 에러가 성공적으로 정정되었습니다.")

    # '&'는 포함하지만 미리 정의된 특정 태그는 포함하지 않는 문자열을 삭제
    # 모든 비식별화 토큰을 정규식에 포함하여 정확한 필터링 적용
    de_id_tokens = '|'.join([
        'address', 'other', 'name', 'affiliation', 'brand', 'location', 
        'online-account', 'company', 'tel-num', 'art', 'anme', 'naem', ' name', 'nama', 
        'affifiation', 'affilation', 'compnay'
    ])
    
    if 'input' in df.columns:
        df = df[~df['input'].str.contains(rf'&(?!{de_id_tokens})', na=False, regex=True)]
    
    print("\n특수 문자 '&'를 포함한 예외적인 행을 삭제했습니다.")
    return df

def construct_tokenized_dataset(dataset, tokenizer, max_length, model_name):
    """입력값(input)에 대하여 토크나이징"""
    print("tokenizer 에 들어가는 데이터 형태")
    print(dataset["input"][:5])

    # RoBERTa 계열 모델은 token_type_ids를 사용하지 않으므로, 모델 이름에 따라 동적으로 설정
    return_token_type_ids = "roberta" not in model_name.lower()

    tokenized_sentences = tokenizer(
        dataset["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_token_type_ids=return_token_type_ids,
    )
    print("tokenizing 된 데이터 형태")
    print("-" * 100)
    print(tokenized_sentences[:5])
    return tokenized_sentences

def prepare_dataset(dataset_dir, tokenizer, max_len, model_name):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data
    train_df = load_data(os.path.join(dataset_dir, "train.csv"))
    valid_df = load_data(os.path.join(dataset_dir, "dev.csv"))
    test_df = load_data(os.path.join(dataset_dir, "test.csv"))
    print("--- data loading Done ---")

    # 데이터 정제
    train_df = clean_data(train_df)
    valid_df = clean_data(valid_df)
    test_df = clean_data(test_df)
    
    # 비식별화 토큰을 special token으로 추가
    special_tokens = ["&location&", "&affiliation&", "&name&", "&company&", "&brand&", 
                      "&art&", "&online-account&", "&address&", "&tel-num&", "&other&"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # split label
    train_label = train_df["output"].values
    valid_label = valid_df["output"].values
    test_label = test_df["output"].values

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_df, tokenizer, max_len, model_name)
    tokenized_valid = construct_tokenized_dataset(valid_df, tokenizer, max_len, model_name)
    tokenized_test = construct_tokenized_dataset(test_df, tokenizer, max_len, model_name)
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    hate_test_dataset = hate_dataset(tokenized_test, test_label)
    print("--- pytorch dataset class Done ---")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_df