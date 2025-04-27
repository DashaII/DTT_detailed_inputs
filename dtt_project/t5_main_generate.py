#!/usr/bin/env python3

from transformers import T5Tokenizer, T5ForConditionalGeneration
from WebNLGData_T5 import WebNLGData
import torch
import configs
from t5_trainer import generate_list_batch
import datasets
from functions import parse_plain_triples_file


def save_to_file(data: list, filename: str):
    with open(filename, "w", encoding='utf-8') as file:
        for item in data:
            file.write("%s\n" % item)


def get_test_sample(data, sample_size: int):
    return data[:sample_size]


def generate_from_pretrained(task_type, prefix, dataset_split, save_to_file_name, subset=None):
    """
    Generates model outputs from a pretrained T5 model for a given dataset split and saves the results to a file.
    Args:
        task_type (str): Task type indicating how to process input data.
        prefix (str): Text prefix to prepend to each input before generation.
        dataset_split (str): Dataset split.
        save_to_file_name (str): Path to save the generated model outputs.
        subset (list, optional): List of indices to subset the dataset before generation (for error analysis).
    """
    t5_tokenizer = T5Tokenizer.from_pretrained(configs.HF_MODEL)
    train_webnlgdata = WebNLGData(split=configs.TRAIN_SPLIT, lang='en', task=task_type,
                                  size=configs.DATASET_SIZE, tokenizer=t5_tokenizer)
    # use the same tokenizer for train and generate (webnlg.tokenizer has special symbols)
    tokenizer = train_webnlgdata.tokenizer

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device type", device)

    # --- GENERATE ---
    finetuned_model = T5ForConditionalGeneration.from_pretrained(configs.MODEL_PATH).to(device)

    if dataset_split == configs.CUSTOM_SPLIT:
        test_dataset = parse_plain_triples_file(configs.CUSTOM_PATH)
    else:
        test_dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=dataset_split)

    # to test on a small dataset
    # test_dataset = get_test_sample(test_dataset, 20)

    # task configs.TASK_BASELINE_PREFIX or configs.TASK_AMR_PREFIX or another task
    generated_output, generated_full = generate_list_batch(test_dataset, prefix, dataset_split, tokenizer, finetuned_model,
                                                     device, subset, batch_size=200)
    save_to_file(generated_output, save_to_file_name)


def generate_baseline(file_name, split=configs.TEST_SPLIT):
    generate_from_pretrained(
        task_type=configs.TASK_TYPE_BASELINE,
        prefix=configs.TASK_BASELINE_PREFIX,
        dataset_split=split,
        save_to_file_name=file_name
    )


def generate_amr(file_name, split=configs.TEST_SPLIT):
    generate_from_pretrained(
        task_type=configs.TASK_TYPE_AMR,
        prefix=configs.TASK_AMR_PREFIX,
        dataset_split=split,
        save_to_file_name=file_name
    )


def generate_amr_enriched(file_name, split=configs.TEST_SPLIT, subset_idx=None):
    generate_from_pretrained(
        task_type=configs.TASK_TYPE_AMR_ENRICHED,
        prefix=configs.TASK_AMR_ENRICHED_PREFIX,
        dataset_split=split,
        save_to_file_name=file_name,
        subset=subset_idx
    )


def generate_plan(file_name, split=configs.TEST_SPLIT):
    generate_from_pretrained(
        task_type=configs.TASK_TYPE_PLAN,
        prefix=configs.TASK_AMR_PREFIX,
        dataset_split=split,
        save_to_file_name=file_name
    )


def generate_plan_enriched(file_name, split=configs.TEST_SPLIT):
    generate_from_pretrained(
        task_type=configs.TASK_TYPE_PLAN_ENRICHED,
        prefix=configs.TASK_PLAN_ENRICHED_PREFIX,
        dataset_split=split,
        save_to_file_name=file_name
    )


if __name__ == '__main__':
    generate_amr_enriched("t5_amr_ENRICHED_results/test_results_amr_enriched_FULL_2nd_model_only_.txt",
                  split=configs.TEST_SPLIT)

