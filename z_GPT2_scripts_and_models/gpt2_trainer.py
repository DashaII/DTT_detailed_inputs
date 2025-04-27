#!/usr/bin/env python3

import configs
from tqdm import tqdm
import torch
from logzero import logger
import os
from WebNLGData import transform_triple


def train(train_data_loader, valid_data_loader, model, optimizer, scheduler, device):
    logger.info('Start training...')
    max_accuracy = 0
    for epoch in range(configs.EPOCHS):
        model.train()
        logger.info(f'\n====== Epoch {epoch+1}/{configs.EPOCHS} Training ======')
        for i, (input_ids, attention_mask, reference_mask) in enumerate(train_data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            reference_mask = reference_mask.to(device)

            # clone input_ids and set labels for context to -100 (ignored by training)
            # from documentation: All labels set to -100 are ignored (masked), the loss is only computed
            # for labels in [0, ..., config.vocab_size]
            labels = input_ids.clone().long().to(device)
            labels[~reference_mask.bool()] = -100
            # forward step
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # backward step
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i > 0 and i % 100 == 0:
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
            model.save_pretrained(os.path.join(os.curdir, configs.MODEL_PATH_FOLDER+str(epoch+1), 'gpt2_webnlg_epoch'+str(epoch+1)))


def evaluate(data_loader, model, device):
    model.eval()

    correct_predictions_sum = 0
    total_predictions = 0
    total_loss = 0

    with torch.no_grad():
        for input_ids, attention_mask, reference_mask in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            reference_mask = reference_mask.to(device)

            labels = input_ids.clone().long().to(device)
            labels[~reference_mask.bool()] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pred = torch.argmax(outputs.logits, dim=-1)
            total_loss += outputs.loss.item()

            # Drop the last prediction
            pred = pred[:, :-1]
            # Shift the labels one position to the left
            labels = labels[:, 1:]

            labels_mask = labels != -100
            correct_predictions = (pred == labels) & labels_mask

            correct_predictions_sum += correct_predictions.sum().item()
            total_predictions += labels_mask.sum().item()

    accuracy = correct_predictions_sum / total_predictions
    valid_loss = total_loss / len(data_loader)
    return accuracy, valid_loss


def generate_one(webnlg_triples, tokenizer, model, device):
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


def generate_list(webnlg_test_data, tokenizer, model, device):
    model.eval()

    outputs_txt = []
    outputs = []
    for idx, item in enumerate(tqdm(webnlg_test_data)):

        inp = ''
        raw_inp = ''
        raw_inp_length = 0
        for triple in item['input']:
            triple_transformed = transform_triple(triple)
            inp += '<|triple|>' + triple_transformed
            raw_inp += triple_transformed + ", "
            raw_inp_length += len(triple_transformed)

        inp = tokenizer(inp, return_tensors='pt', padding=True)
        input_ids = inp['input_ids'].to(device)
        attention_mask = inp['attention_mask'].to(device)

        output = model.generate(input_ids, attention_mask=attention_mask, max_length=500, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(output[0], skip_special_tokens=True)

        outputs_txt.append(str(idx+1) + " triple > " + raw_inp)
        outputs_txt.append(str(idx+1) + " text   > " + output[raw_inp_length:])
        outputs.append(output[raw_inp_length:])

    return outputs, outputs_txt
