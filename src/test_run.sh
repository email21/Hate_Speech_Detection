#!/bin/bash

# python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 1e-5 --batch_size 16 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 2e-5 --batch_size 16 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 3e-5 --batch_size 16 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 1e-5 --batch_size 32 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 2e-5 --batch_size 32 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 3e-5 --batch_size 32 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 1e-5 --batch_size 64 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 2e-5 --batch_size 64 --epochs 10
python main.py --model_name "klue/roberta-large" --run_name "roberta-large-test" --lr 3e-5 --batch_size 64 --epochs 10
