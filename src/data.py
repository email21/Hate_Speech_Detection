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

def load_data(dataset_name, split):
    """HuggingFace에서 데이터셋 로드 → pandas DataFrame 반환"""
    from datasets import load_dataset
    
    try:
        # HF Dataset 로드
        hf_dataset = load_dataset(dataset_name, split=split)
        
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


def prepare_dataset(dataset_name, tokenizer, max_len, model_name):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data
    # train_dataset = load_data(os.path.join(dataset_dir, "train.csv")) 
    # valid_dataset = load_data(os.path.join(dataset_dir, "dev.csv"))
    # test_dataset = load_data(os.path.join(dataset_dir, "test.csv"))
    # print("--- data loading Done ---")
    
    # HuggingFace에서 데이터 로드
    train_dataset = load_data(dataset_name, "train")
    valid_dataset = load_data(dataset_name, "validation") 
    test_dataset = load_data(dataset_name, "test")
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

def prepare_dataset_for_cpt(dataset_name, tokenizer, max_length):
    """추가 사전 훈련(CPT)을 위한 데이터셋 준비"""
    #from datasets import load_dataset
    
    print(f"CPT용 데이터 로드: {dataset_name}")
    
    # 훈련 데이터만 사용하여 모든 텍스트를 하나의 코퍼스로 만듭니다.
    # unlabeled 데이터가 있다면 함께 사용하는 것이 좋습니다.
    dataset = load_dataset(dataset_name, split="train")
    
    # 텍스트가 없는 라인 필터링
    #dataset = dataset.filter(lambda x: x['input'] is not None and len(x['input']) > 0)
    
    def tokenize_function(examples):
        return tokenizer(examples["input"], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=dataset.column_names
    )

    block_size = tokenizer.model_max_length
    if block_size > 1024:
        block_size = 1024
        
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
    print("CPT용 데이터셋 준비 완료.")
    return lm_dataset