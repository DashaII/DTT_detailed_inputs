#!/usr/bin/env python3

from torch.utils.data import Dataset
import datasets
import transformers
import configs
import re


def transform_triple(triple):
    res_triple = triple.replace("_", " ")
    triple_split = res_triple.split("|")
    # camel_to_space = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', triple_split[1]))
    camel_to_space = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', triple_split[1])
    triple_split[1] = camel_to_space.lower()
    if triple_split[2][-2:] == '.0' and triple_split[2][:-2].strip().isdigit():
        triple_split[2] = triple_split[2][:-2]
    res_triple = triple_split[0] + "|" + triple_split[1] + "|" + triple_split[2]

    return res_triple


class WebNLGData(Dataset):
    def __init__(self, split='train', lang=configs.LANG, size=None, tokenizer=None):
        self.split = split
        self.lang = lang
        dataset = datasets.load_dataset(path='GEM/web_nlg', name=lang, split=split)

        data = []
        triples = []
        for idx, item in enumerate(dataset):
            if size is None:
                example = ''
                for triple in item['input']:
                    transformed_triple = transform_triple(triple)
                    example += '<|triple|>' + transformed_triple
                triples.append(example)
                example += '<|target|>' + item['target'] + '<|endoftext|>'
                data.append(example)
            else:
                if 100 <= idx < 100 + size:
                    example = ''
                    for triple in item['input']:
                        transformed_triple = transform_triple(triple)
                        example += '<|triple|>' + transformed_triple
                    triples.append(example)
                    example += '<|target|>' + item['target'] + '<|endoftext|>'
                    data.append(example)
        self.data = data
        if tokenizer is None:
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        else:
            self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(
            {"pad_token": "<pad>",
             "additional_special_tokens": configs.SPECIAL_TOKENS})

        self.data_encoded = self.tokenizer(self.data, truncation=True, padding=True, return_tensors='pt')
        triples_encoded = self.tokenizer(triples, truncation=True, padding=True, return_tensors='pt')

        self.input_ids = self.data_encoded['input_ids']
        self.attention_mask = self.data_encoded['attention_mask']

        # reference mask distinguishes reference (target) from input triples and padding
        # example: 000001111000, where 00000 is masked input, 000 is padding
        triples_attention_mask = triples_encoded['attention_mask']
        triple_len = triples_attention_mask.shape[1]
        self.reference_mask = self.attention_mask.clone()
        self.reference_mask[:, :triple_len][triples_attention_mask == 1] = 0

        print("split: ", split, "len:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.reference_mask[idx]

