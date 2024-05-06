#!/usr/bin/env python3

from transformers import GPT2LMHeadModel, get_scheduler
from WebNLGData import WebNLGData
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import configs
from trainer import train, generate_list, generate_one
import datasets


# TEST EXAMPLE OF GPT2 WITHOUT FINETUNING
# text1 = "who is the best"
# text2 = "why birds don't"
#
# print('tokenized :', tokenizer(text1, return_tensors="pt"))
#
# generated = model.generate(**tokenizer(text1, return_tensors="pt"))
# print('generated :', generated)
# decoded = tokenizer.decode(generated[0])
#
# print('decoded :', decoded)


def save_to_file(data: list, filename: str):
    with open(filename, "w", encoding='utf-8') as file:
        for item in data:
            file.write("%s\n" % item)


def get_test_sample(data, sample_size: int):
    result = []
    for i, item in enumerate(data):
        if i < sample_size:
            result.append(item)
        else:
            break
    return result


if __name__ == '__main__':
    train_webnlgdata = WebNLGData(split=configs.TRAIN_SPLIT, lang='en', size=configs.DATASET_SIZE)
    valid_webnlgdata = WebNLGData(split=configs.VALID_SPLIT, lang='en', size=configs.DATASET_SIZE)

    train_dataloader = DataLoader(train_webnlgdata, batch_size=configs.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_webnlgdata, batch_size=configs.BATCH_SIZE, shuffle=True)

    # use the same tokenizer for train and generate
    # webnlg.tokenizer has special symbols
    tokenizer = train_webnlgdata.tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device type", device)
    # model
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    # len(tokenizer) already includes special tokens and padding
    print("tokenizer len", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=configs.LEARNING_RATE)
    num_training_steps = configs.EPOCHS * len(train_dataloader)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=num_training_steps)

    # --- TRAIN ---
    # train(train_data_loader=train_dataloader, valid_data_loader=valid_dataloader, model=model, optimizer=optimizer,
    #       scheduler=scheduler, device=device)

    # --- GENERATE ---
    finetuned_model = GPT2LMHeadModel.from_pretrained(configs.MODEL_PATH)

    test_triple1 = {
        'gem_id': 'web_nlg_ru-validation-50',
        'gem_parent_id': 'web_nlg_ru-validation-50',
        'input': ['Perth | country | Australia'],
        'target': 'Перт находится в Австралии.',
        'references': ['Перт находится в Австралии.', 'Перт находится в Австралии.'],
    }
    test_triple2 = {
        'gem_id': 'web_nlg_ru-validation-100',
        'gem_parent_id': 'web_nlg_ru-validation-100',
        'input': ['Sheldon Moldoff | award | Inkpot Award'],
        'target': 'Премию Inkpot получил Шелдон Молдофф.',
        'references': ['Премию Inkpot получил Шелдон Молдофф.', 'Шелдон Молдофф получил премию Inkpot.'],
    }

    generated = generate_one(test_triple2, tokenizer, finetuned_model, device)
    print(generated)

    test_dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=configs.TEST_SPLIT)
    # test_dataset = get_test_sample(test_dataset, 20)
    generated_output, generated_full = generate_list(test_dataset, tokenizer, finetuned_model, device)

    save_to_file(generated_full, "results/results_generated_full_12.txt")
    save_to_file(generated_output, "results/results_generated_12.txt")
