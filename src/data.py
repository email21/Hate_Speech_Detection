import os
import pandas as pd
import torch
from datasets import load_dataset

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


# def load_data(dataset_dir):
#     """csv file을 dataframe으로 load"""
#     dataset = pd.read_csv(dataset_dir)
#     print("dataframe 의 형태")
#     print("-" * 100)
#     print(dataset.head())
#     return dataset

def load_data(dataset_name, split, revision):
    """HuggingFace에서 데이터셋 로드 → pandas DataFrame 반환"""
    from datasets import load_dataset
    
    try:
        # HF Dataset 로드
        hf_dataset = load_dataset(dataset_name, revision=revision, split=split)
        
        # pandas DataFrame으로 변환
        dataset = hf_dataset.to_pandas()
        
        print(f"dataframe 의 형태 ({split})")
        print("-" * 100)
        print(dataset.head())
        return dataset
        
    except Exception as e:
        print(f"데이터 로드 에러: {e}")
        return None
    
def construct_tokenized_dataset(dataset, tokenizer, max_length, model_name):
    """입력값(input)에 대하여 토크나이징"""
    print("tokenizer 에 들어가는 데이터 형태")
    print(dataset["input"][:5])

    # RoBERTa 계열 모델은 token_type_ids를 사용하지 않으므로, 모델 이름에 따라 동적으로 설정
    return_token_type_ids = "roberta" not in model_name.lower()
    
    tokenized_senetences = tokenizer(
        dataset["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_token_type_ids=return_token_type_ids,
        # return_token_type_ids=False,  # BERT 이후 모델(RoBERTa 등) 사용할때 False
    )
    print("tokenizing 된 데이터 형태")
    print("-" * 100)
    print(tokenized_senetences[:5])
    return tokenized_senetences


def prepare_dataset(dataset_name, tokenizer, max_len, model_name, revision):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data
    # train_dataset = load_data(os.path.join(dataset_dir, "train.csv")) 
    # valid_dataset = load_data(os.path.join(dataset_dir, "dev.csv"))
    # test_dataset = load_data(os.path.join(dataset_dir, "test.csv"))
    # print("--- data loading Done ---")
    
    # HuggingFace에서 데이터 로드
    train_dataset = load_data(dataset_name, "train", revision=revision)
    valid_dataset = load_data(dataset_name, "validation", revision=revision) 
    test_dataset = load_data(dataset_name, "test", revision=revision)
    print("--- data loading Done ---")

    # split label
    train_label = train_dataset["output"].values
    valid_label = valid_dataset["output"].values
    test_label = test_dataset["output"].values

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_dataset, tokenizer, max_len, model_name)
    tokenized_valid = construct_tokenized_dataset(valid_dataset, tokenizer, max_len, model_name)
    tokenized_test = construct_tokenized_dataset(test_dataset, tokenizer, max_len, model_name)
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    hate_test_dataset = hate_dataset(tokenized_test, test_label)
    print("--- pytorch dataset class Done ---")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset

def prepare_kfold_dataset(dataset_name, revision):
    """K-Fold를 위해 전체 train 데이터와 test 데이터를 불러오는 함수"""
    # HuggingFace에서 전체 학습 데이터 로드
    # K-Fold는 학습 데이터를 나누는 것이므로, validation은 따로 불러오지 않습니다.
    full_train_dataset = load_data(dataset_name, "train", revision=args.revision)
    test_dataset = load_data(dataset_name, "test", revision=args.revision)

    if full_train_dataset is None or test_dataset is None:
        raise ValueError("K-Fold용 데이터셋 로딩에 실패했습니다.")

    print("--- K-Fold data loading Done ---")
    return full_train_dataset, test_dataset