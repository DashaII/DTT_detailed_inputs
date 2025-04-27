#!/usr/bin/env python3

import configs
from tqdm import tqdm
import torch
from logzero import logger
import os
from WebNLGData_T5 import transform_triple


def train(train_data_loader, valid_data_loader, model, optimizer, scheduler, device):
    """
    Trains the given model using the provided training and validation data loaders.
    The best model (based on validation accuracy) is saved after each epoch.
    """
    logger.info('Start training...')
    max_accuracy = 0
    for epoch in range(configs.EPOCHS):
        model.train()
        logger.info(f'\n====== Epoch {epoch+1}/{configs.EPOCHS} Training ======')
        for i, (input_ids, attention_mask, target_ids) in enumerate(train_data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            # forward step
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            # backward step
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i > 0 and i % 50 == 0:
                logger.info(f'loss: {output["loss"]}')
            if i > 0 and i % 2000 == 0:
                accuracy, loss = evaluate(valid_data_loader, model, device)
                logger.info(f'\nVALID Accuracy: {accuracy}')
                logger.info(f'\nVALID Loss: {loss}')

        valid_accuracy, valid_loss = evaluate(valid_data_loader, model, device)
        logger.info(f'\nVALID Accuracy after epoch {epoch + 1}: {valid_accuracy}')
        logger.info(f'\nVALID Loss after epoch {epoch + 1}: {valid_loss}')
        train_accuracy, train_loss = evaluate(train_data_loader, model, device)
        logger.info(f'\nTRAIN Accuracy after epoch {epoch + 1}: {train_accuracy}')

        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            model.save_pretrained(os.path.join(os.curdir, configs.MODEL_PATH_FOLDER+str(epoch+1), 't5_webnlg_epoch'+str(epoch+1)))


def evaluate(data_loader, model, device):
    """
    Evaluates the given model on the provided data loader. Computes the accuracy and average loss across the dataset.
    """
    correct_predictions_sum = 0
    total_predictions = 0
    total_loss = 0

    with torch.no_grad():
        for input_ids, attention_mask, target_ids in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            pred = torch.argmax(outputs.logits, dim=-1)
            total_loss += outputs.loss.item()

            labels_mask = target_ids != -100
            correct_predictions = (pred == target_ids) & labels_mask

            correct_predictions_sum += correct_predictions.sum().item()
            total_predictions += labels_mask.sum().item()

    accuracy = correct_predictions_sum / total_predictions
    valid_loss = total_loss / len(data_loader)
    return accuracy, valid_loss


def generate_one(webnlg_triples, tokenizer, model, device):
    """
    Generates a text output from a single set of WebNLG triples.
    """
    model.eval()

    inp = ''
    raw_inp = ''
    raw_inp_length = 0
    for triple in webnlg_triples['input']:
        triple_transformed = transform_triple(triple)
        inp += '<|triple|>' + triple_transformed
        raw_inp += triple_transformed + ", "
        raw_inp_length += len(triple_transformed)

    inp = tokenizer(inp, return_tensors='pt')
    input_ids = inp['input_ids'].to(device)
    attention_mask = inp['attention_mask'].to(device)

    output = model.generate(input_ids, attention_mask=attention_mask, max_length=200,
                            pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    triple = "triple > " + raw_inp
    output = "text   > " + output[raw_inp_length:]

    return triple, output


def generate_list(webnlg_test_data, prefix, split, tokenizer, model, device, subset=None):
    """
    Generates text outputs for a list of WebNLG test data entries.
    Handles optionally enriched structures (AMR or flat plan) and generates texts individually for each input.
    Args:
        webnlg_test_data (list): List of WebNLG entries.
        prefix (str): Prefix to prepend.
        split (str): Dataset split.
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding and decoding text.
        model (torch.nn.Module): Model for generation.
        device (torch.device): Device to run computations on.
        subset (list, optional): List of indices to process; if None, processes all.
    Returns: tuple: (list of generated texts (str), list of formatted outputs (str))
    """
    model.eval()

    outputs_txt = []
    outputs = []

    if prefix == configs.TASK_AMR_ENRICHED_PREFIX:
        if split == configs.CUSTOM_SPLIT:
            with open(f"{configs.AMR_GENERATED_SIMPLIFIED_CUSTOM}.txt", "r", encoding='utf-8') as file:
                generated_structures = file.readlines()
        else:
            with open(f"{configs.AMR_GENERATED_SIMPLIFIED}{split}.txt", "r", encoding='utf-8') as file:
                generated_structures = file.readlines()
    elif prefix == configs.TASK_PLAN_ENRICHED_PREFIX:
        with open(f"{configs.PLAN_GENERATED}{split}.txt", "r", encoding='utf-8') as file:
            generated_structures = file.readlines()

    subset_idx = subset if subset is not None else []
    for idx, item in enumerate(tqdm(webnlg_test_data)):
        if len(subset_idx) != 0 and idx not in subset_idx:
            continue

        inp = prefix
        raw_inp = ''
        raw_structure = ''
        item_input = item if split == configs.CUSTOM_SPLIT else item['input']

        for triple in item_input:
            triple_transformed = transform_triple(triple)
            inp += '<|triple|>' + triple_transformed
            raw_inp += triple_transformed + ", "
        if prefix == configs.TASK_AMR_ENRICHED_PREFIX or prefix == configs.TASK_PLAN_ENRICHED_PREFIX:
            inp += '<|structure|>' + generated_structures[idx]
            raw_structure = generated_structures[idx].strip()

        # Tokenize the input and move to the correct device
        # inp = tokenizer(inp, return_tensors='pt', padding=True)
        inp = tokenizer(inp, return_tensors='pt')
        input_ids = inp['input_ids'].to(device)
        attention_mask = inp['attention_mask'].to(device)

        output = model.generate(input_ids, attention_mask=attention_mask, max_length=500)
        output = tokenizer.decode(output[0], skip_special_tokens=True)

        outputs_txt.append(str(idx+1) + " triple    > " + raw_inp)
        if prefix == configs.TASK_AMR_ENRICHED_PREFIX:
            outputs_txt.append(str(idx + 1) + " structure > " + raw_structure)
        outputs_txt.append(str(idx+1) + " text      > " + output)
        outputs.append(output)

    return outputs, outputs_txt


def generate_list_batch(webnlg_test_data, prefix, split, tokenizer, model, device, subset=None, batch_size=8):
    """
    Generates text outputs for a list of WebNLG test data entries in batches.
    Similar to `generate_list` but processes inputs in batches.
    Handles optional structure enrichment if required.
    Args: batch_size (int, optional): Batch size for generation (default 8).
    Returns: tuple: (list of generated texts (str), list of formatted outputs (str))
    """
    model.eval()

    outputs_txt = []
    outputs = []

    # load generated structures if needed
    if prefix == configs.TASK_AMR_ENRICHED_PREFIX:
        filename = f"{configs.AMR_GENERATED_SIMPLIFIED_CUSTOM}.txt" if split == configs.CUSTOM_SPLIT else f"{configs.AMR_GENERATED_SIMPLIFIED}{split}.txt"
        with open(filename, "r", encoding='utf-8') as file:
            generated_structures = file.readlines()
    elif prefix == configs.TASK_PLAN_ENRICHED_PREFIX:
        with open(f"{configs.PLAN_GENERATED}{split}.txt", "r", encoding='utf-8') as file:
            generated_structures = file.readlines()
    else:
        generated_structures = ["" for _ in range(len(webnlg_test_data))]

    subset_idx = set(subset) if subset is not None else set(range(len(webnlg_test_data)))

    batch_inputs = []
    batch_raw_inps = []
    batch_structures = []
    batch_indices = []

    for idx, item in enumerate(tqdm(webnlg_test_data)):
        if idx not in subset_idx:
            continue

        inp = prefix
        raw_inp = ''
        raw_structure = ''
        item_input = item if split == configs.CUSTOM_SPLIT else item['input']

        for triple in item_input:
            triple_transformed = transform_triple(triple)
            inp += '<|triple|>' + triple_transformed
            raw_inp += triple_transformed + ", "

        if prefix in [configs.TASK_AMR_ENRICHED_PREFIX, configs.TASK_PLAN_ENRICHED_PREFIX]:
            inp += '<|structure|>' + generated_structures[idx]
            raw_structure = generated_structures[idx].strip()

        batch_inputs.append(inp)
        batch_raw_inps.append(raw_inp)
        batch_structures.append(raw_structure)
        batch_indices.append(idx + 1)

        # process the batch
        if len(batch_inputs) >= batch_size:
            encoded = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                output_ids = model.generate(**encoded, max_length=500)
            batch_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for i, output in enumerate(batch_outputs):
                outputs_txt.append(f"{batch_indices[i]} triple    > {batch_raw_inps[i]}")
                if prefix == configs.TASK_AMR_ENRICHED_PREFIX:
                    outputs_txt.append(f"{batch_indices[i]} structure > {batch_structures[i]}")
                outputs_txt.append(f"{batch_indices[i]} text      > {output}")
                outputs.append(output)

            # clear batch
            batch_inputs.clear()
            batch_raw_inps.clear()
            batch_structures.clear()
            batch_indices.clear()

    # remaining examples
    if batch_inputs:
        encoded = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_length=500)
        batch_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i, output in enumerate(batch_outputs):
            outputs_txt.append(f"{batch_indices[i]} triple    > {batch_raw_inps[i]}")
            if prefix == configs.TASK_AMR_ENRICHED_PREFIX:
                outputs_txt.append(f"{batch_indices[i]} structure > {batch_structures[i]}")
            outputs_txt.append(f"{batch_indices[i]} text      > {output}")
            outputs.append(output)

    return outputs, outputs_txt
