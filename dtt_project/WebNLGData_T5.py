#!/usr/bin/env python3

from torch.utils.data import Dataset
import datasets
import transformers
import configs
import re
import random
import json


def transform_triple(triple, subj_obj_swap=False):
    """
    Cleans and transforms a triple string by replacing underscores, removing quotes,
    normalizing camel case predicates, and optionally swapping subject and object.
    """
    res_triple = triple.replace("_", " ").replace('"', '').replace("'", '')
    triple_split = res_triple.split("|")
    camel_to_space = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', triple_split[1])
    triple_split[1] = camel_to_space.lower()
    if triple_split[2][-2:] == '.0' and triple_split[2][:-2].strip().isdigit():
        triple_split[2] = triple_split[2][:-2]

    if subj_obj_swap:
        triple_split[0], triple_split[2] = triple_split[2].strip(), triple_split[0].strip()
    else:
        triple_split[0], triple_split[2] = triple_split[0].strip(), triple_split[2].strip()

    return f"{triple_split[0]} | {triple_split[1].strip()} | {triple_split[2]}"


def transform_webnlg_data(task, split, dataset, size, shuffle_triples, skip_idx):
    """
    Transforms WebNLG dataset into model-ready inputs and targets based on the task type.
    """
    inputs, targets = [], []

    def create_example(item, prefix, swap_list=None, shuffle_triples=False):
        example = prefix
        if not shuffle_triples:
            if swap_list is None:  # swap is possible for plan structures only
                for triple in item['input']:
                        transformed_triple = transform_triple(triple)
                        example += '<|triple|>' + transformed_triple
            else:
                for idx, triple in enumerate(item['input']):
                        transformed_triple = transform_triple(triple, swap_list[idx])
                        example += '<|triple|>' + transformed_triple
        else:
            indices = list(range(len(item['input'])))
            random.shuffle(indices)  # shuffle the indices
            for i in indices:
                triple = item['input'][i]
                transformed_triple = transform_triple(triple)
                example += '<|triple|>' + transformed_triple
        return example

    if task == configs.TASK_TYPE_BASELINE:
        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            if idx in skip_idx:
                continue
            example = create_example(item, configs.TASK_BASELINE_PREFIX)
            inputs.append(example)
            targets.append(item['target'])

    elif task == configs.TASK_TYPE_AMR:
        with open(f"{configs.AMR_PARSER_SIMPLIFIED}{split}.txt", "r", encoding='utf-8') as file:
            amr_flat = file.readlines()

        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            if idx in skip_idx:
                continue
            # if there is just one triple, don't shuffle
            if shuffle_triples is None or len(item['input']) == 1:
                example = create_example(item, configs.TASK_AMR_PREFIX)
                inputs.append(example)
                targets.append(amr_flat[idx])
            else:
                for _ in range(shuffle_triples):
                    example = create_example(item, configs.TASK_AMR_PREFIX, True)
                    inputs.append(example)
                    targets.append(amr_flat[idx])

    elif task == configs.TASK_TYPE_AMR_ENRICHED:
        with open(f"{configs.AMR_GENERATED_SIMPLIFIED}{split}.txt", "r", encoding='utf-8') as file:
            amr_flat_generated = file.readlines()

        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            if idx in skip_idx:
                continue
            # if there is just one triple, don't shuffle
            if shuffle_triples is None or len(item['input']) == 1:
                example = create_example(item, configs.TASK_AMR_ENRICHED_PREFIX)
                example += '<|structure|>' + amr_flat_generated[idx]
                inputs.append(example)
                targets.append(item['target'])
            else:
                for _ in range(shuffle_triples):
                    example = create_example(item, configs.TASK_AMR_ENRICHED_PREFIX, None, True)
                    example += '<|structure|>' + amr_flat_generated[idx]
                    inputs.append(example)
                    targets.append(item['target'])

    elif task == configs.TASK_TYPE_PLAN:
        with open(f"{configs.PLAN_PARSER}{split}.txt", "r", encoding='utf-8') as file:
            plan = file.readlines()
        with open(f"{configs.PLAN_SWAP_TRIPE}{split}.json", "r", encoding='utf-8') as file:
            swap_dict = json.load(file)

        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            if idx in skip_idx:
                continue
            example = create_example(item, configs.TASK_AMR_PREFIX, swap_list=swap_dict[str(idx)])
            inputs.append(example)
            targets.append(plan[idx])

    elif task == configs.TASK_TYPE_PLAN_ENRICHED:
        with open(f"{configs.PLAN_GENERATED}{split}.txt", "r", encoding='utf-8') as file:
            plan_generated = file.readlines()

        for idx, item in enumerate(dataset):
            if size is not None and not (100 <= idx < 100 + size):
                continue
            if idx in skip_idx:
                continue
            example = create_example(item, configs.TASK_AMR_ENRICHED_PREFIX)
            example += '<|structure|>' + plan_generated[idx]
            inputs.append(example)
            targets.append(item['target'])

    return inputs, targets


class WebNLGData(Dataset):
    """
    PyTorch Dataset for loading and tokenizing WebNLG data for generation tasks.
    Args:
        split: Data split.
        lang: Language.
        task: Type of task to prepare the data for.
        shuffle_triples: Number of times to shuffle triples.
        skip_idx (optional): Indices to skip.
        size (optional): Size limit for data subset.
        tokenizer (optional): Tokenizer to use. If None, GPT2Tokenizer is used.
    """
    def __init__(self, split='train', lang=configs.LANG, task=configs.TASK_TYPE_BASELINE, shuffle_triples=None,
                 skip_idx=None, size=None, tokenizer=None):
        self.split = split
        self.lang = lang
        dataset = datasets.load_dataset(path='GEM/web_nlg', name=lang, split=split)

        skip_idx = skip_idx if skip_idx is not None else []
        self.inputs, self.targets = transform_webnlg_data(task, split, dataset, size, shuffle_triples, skip_idx)

        if tokenizer is None:
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        else:
            self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(
            {"pad_token": "<pad>",
             "additional_special_tokens": configs.SPECIAL_TOKENS})

        self.inputs_encoded = self.tokenizer(self.inputs, truncation=True, padding=True, return_tensors='pt')
        self.targets_encoded = self.tokenizer(self.targets, truncation=True, padding=True, return_tensors='pt')

        self.input_ids = self.inputs_encoded['input_ids']
        self.attention_mask = self.inputs_encoded['attention_mask']
        self.labels = self.targets_encoded['input_ids']

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        self.labels[self.labels == tokenizer.pad_token_id] = -100

        print("split: ", split, "len:", len(self.inputs))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

