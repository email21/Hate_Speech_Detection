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
    # 토큰 오타 정정
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
    
    print("\n'input' 열의 오타 정정 완료")

    # '&'가 포함된 행 중, 유효한 태그가 없는 행만 삭제하는 로직으로 수정
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
    
    original_count = len(df)
    # 5. 해당 행들을 제외하고 데이터프레임을 새로 만듦
    df = df[~rows_to_delete_mask].copy()
    deleted_count = original_count - len(df)
    
    if deleted_count > 0:
        print(f"유효한 태그 없이 '&'를 포함한 예외적인 행 {deleted_count}개를 삭제했습니다.")
    
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
    """학습(train)과 검증(validation)을 위한 데이터셋을 준비"""
    # load_data
    train_df = load_data(os.path.join(dataset_dir, "train.csv"))
    valid_df = load_data(os.path.join(dataset_dir, "dev.csv"))
    print("--- data loading Done ---")

    # 학습 및 검증 데이터에 대해서만 정제(행 삭제) 수행
    train_df = clean_data(train_df)
    valid_df = clean_data(valid_df)
    
    # 비식별화 토큰을 special token으로 추가
    special_tokens = ["&location&", "&affiliation&", "&name&", "&company&", "&brand&", 
                      "&art&", "&online-account&", "&address&", "&tel-num&", "&other&"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print("추가된 스페셜 토큰:", tokenizer.special_tokens_map)
    
    # split label
    train_label = train_df["output"].values
    valid_label = valid_df["output"].values

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_df, tokenizer, max_len, model_name)
    tokenized_valid = construct_tokenized_dataset(valid_df, tokenizer, max_len, model_name)
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    print("--- pytorch dataset class Done ---")

    # return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_df
    return hate_train_dataset, hate_valid_dataset, None, None