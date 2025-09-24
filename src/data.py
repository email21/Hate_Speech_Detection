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
        hf_dataset = load_dataset(dataset_name, split=split, revision=revision)

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
    train_dataset = load_data(dataset_name, "train", revision)
    valid_dataset = load_data(dataset_name, "validation", revision)
    test_dataset = load_data(dataset_name, "test", revision)
    print("--- data loading Done ---")

    # split label
    train_label = train_dataset["output"].values
    valid_label = valid_dataset["output"].values
    test_label = test_dataset["output"].values

    # [수정] test_dataset에 'output' 컬럼이 없을 경우를 대비합니다.
    # 이렇게 하면 원본 파일을 건드리지 않고 코드의 안정성을 높일 수 있습니다.
    if "output" in test_dataset.columns:
        test_label = test_dataset["output"].values
    else:
        # 'output' 컬럼이 없으면, 임시로 0으로 채웁니다.
        # 이 값은 추론 시에는 사용되지 않으므로 어떤 값이든 상관없습니다.
        test_label = [0] * len(test_dataset)

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(
        train_dataset, tokenizer, max_len, model_name
    )
    tokenized_valid = construct_tokenized_dataset(
        valid_dataset, tokenizer, max_len, model_name
    )
    tokenized_test = construct_tokenized_dataset(
        test_dataset, tokenizer, max_len, model_name
    )
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    hate_test_dataset = hate_dataset(tokenized_test, test_label)
    print("--- pytorch dataset class Done ---")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset
