import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    # calculate f1 score using sklearn's function
    f1 = f1_score(labels, preds, average="micro")

    return {
        "accuracy": acc,
        "f1": f1,
    }


# 아래는 평가지표에 따라 compute_metrics를 수정한 것.(성능이 좋지 않다..)
# import numpy as np


# def compute_metrics(eval_pred):
#     #     """validation을 위한 metrics function"""
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1)  # argmax 기준으로 예측

#     TP = np.sum((preds == 1) & (labels == 1))
#     FP = np.sum((preds == 1) & (labels == 0))
#     FN = np.sum((preds == 0) & (labels == 1))

#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#     f1 = (
#         2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#     )

#     return {"f1_micro": f1}
