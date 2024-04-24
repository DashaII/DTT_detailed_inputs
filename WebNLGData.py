#!/usr/bin/env python3

from torch.utils.data import Dataset
import datasets
import transformers
import configs


class WebNLGData(Dataset):
    def __init__(self, split='train', lang=configs.LANG, size=None):
        self.split = split
        self.lang = lang
        dataset = datasets.load_dataset(path='GEM/web_nlg', name=lang, split=split)

        data = []
        triples = []
        for idx, item in enumerate(dataset):
            if size is None:
                example = ''
                for triple in item['input']:
                    # ['Punjab,_Pakistan | leaderTitle | Provincial_Assembly_of_the_Punjab'] ->
                    # ['Punjab, Pakistan | leaderTitle | Provincial Assembly of the Punjab']
                    triple = triple.replace("_", " ")
                    example += '<|triple|>' + triple
                triples.append(example)
                example += '<|target|>' + item['target'] + '<|endoftext|>'
                data.append(example)
            else:
                if 100 <= idx < 100 + size:
                    example = ''
                    for triple in item['input']:
                        triple = triple.replace("_", " ")
                        example += '<|triple|>' + triple
                    triples.append(example)
                    example += '<|target|>' + item['target'] + '<|endoftext|>'
                    data.append(example)
        self.data = data

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
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
        # print(self.data[len(self.data)-1])
        # print(self.input_ids[len(self.data)-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.reference_mask[idx]

    # def parse_data_into_examples(self, item):
    #     examples = []
    #
    #     input = item['input']
    #     target = item['target']
    #
    #     example = ''
    #     for triple in input:
    #         example += '<|triple|>' + triple
    #     example += '<|target|>' + target + '<|endoftext|>'
    #
    #     return examples
