#!/usr/bin/env python3

from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from WebNLGData_T5 import WebNLGData
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import configs
from t5_trainer import train


def save_to_file(data: list, filename: str):
    with open(filename, "w", encoding='utf-8') as file:
        for item in data:
            file.write("%s\n" % item)


def get_test_sample(data, sample_size: int):
    result = []
    for i, item in enumerate(data):
        if i < sample_size:
            result.append(item)
        elif i >= sample_size:
            break
    return result


def train_model(task_type, shuffle_triples=None, skip_idx=None):
    """
    Train model
    :param task_type: train task type (e.g. baseline, arm)
    :param shuffle_triples: number of iterations for one example. The iteration means in case of more than one triple
    in an example, the tiples are shuffled shuffle_triples times and all the shuffled example are used for training.
    :param skip_idx: list of dev set indices to be skipped during training
    """
    t5_tokenizer = T5Tokenizer.from_pretrained(configs.HF_MODEL)
    train_webnlgdata = WebNLGData(split=configs.TRAIN_SPLIT, lang='en', task=task_type, shuffle_triples=shuffle_triples,
                                  skip_idx=skip_idx, size=configs.DATASET_SIZE, tokenizer=t5_tokenizer)
    valid_webnlgdata = WebNLGData(split=configs.VALID_SPLIT, lang='en', task=task_type, shuffle_triples=shuffle_triples,
                                  skip_idx=skip_idx, size=configs.DATASET_SIZE, tokenizer=t5_tokenizer)

    train_dataloader = DataLoader(train_webnlgdata, batch_size=configs.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_webnlgdata, batch_size=configs.BATCH_SIZE, shuffle=True)

    # use the same tokenizer for train and generate (webnlg.tokenizer has special symbols)
    tokenizer = train_webnlgdata.tokenizer

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device type", device)
    # model
    model = T5ForConditionalGeneration.from_pretrained(configs.HF_MODEL).to(device)
    # len(tokenizer) already includes special tokens and padding
    print("tokenizer len", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=configs.LEARNING_RATE)
    num_training_steps = configs.EPOCHS * len(train_dataloader)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=num_training_steps)
    # --- TRAIN ---
    train(train_data_loader=train_dataloader, valid_data_loader=valid_dataloader, model=model, optimizer=optimizer,
          scheduler=scheduler, device=device)


if __name__ == '__main__':
    # train_model(task_type=configs.TASK_TYPE_AMR, shuffle_triples=3)
    # train_model(task_type=configs.TASK_TYPE_AMR_ENRICHED, shuffle_triples=3)
    train_model(task_type=configs.TASK_TYPE_PLAN_ENRICHED)
