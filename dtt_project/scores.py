#!/usr/bin/env python3

import sacrebleu
from bleurt import score
import configs
import datasets
import numpy as np
from tqdm import tqdm
import warnings
from evaluate import load

warnings.filterwarnings("ignore")

TEST_RESULTS_FILES = [
    't5_amr_enriched_results/test_results_amr_enriched_0_6.txt',
    't5_amr_enriched_results/test_results_amr_enriched_1_10.txt',
    't5_amr_enriched_results/test_results_amr_enriched_1_5.txt',
    't5_amr_enriched_results/test_results_amr_enriched_3_6.txt',
    't5_amr_enriched_results/test_results_amr_enriched_4_4.txt',
    't5_results/test_results_generated_6_6.txt',

    't5_amr_enriched_results/test_results_amr_enriched_shuffled_1_12.txt',
    't5_amr_enriched_results/test_results_amr_enriched_shuffled_1_7.txt',
    't5_amr_enriched_results/test_results_amr_enriched_shuffled_1_1.txt',

    "t5_amr_ENRICHED_results/test_results_amr_enriched_FULL_FINAL_amrs_2_3.txt",
    "t5_amr_ENRICHED_results/test_results_amr_enriched_FULL_2nd_model_only.txt"

    't5_plan_ENRICHED_results/test_plan_trivial_full_pred_1_5.txt',
    't5_plan_ENRICHED_results/test_plan_trivial_full_NO_pred_2_7.txt',
    't5_plan_ENRICHED_results/test_plan_trivial_short_pred_3_5.txt',
    't5_plan_ENRICHED_results/test_plan_trivial_short_4_5.txt',

    't5_plan_ENRICHED_results/ENRICHED_plan_pred_short_results_4_test.txt',
    't5_plan_ENRICHED_results/ENRICHED_plan_pred_results_5_test.txt',
    't5_plan_ENRICHED_results/ENRICHED_plan_short_results_5_test.txt',
    't5_plan_ENRICHED_results/ENRICHED_plan_results_6_test.txt',
    't5_plan_ENRICHED_results/ENRICHED_plan_results_6_2_test.txt'
]
TEST_RESULTS_FILES_SEEN = [
    't5_results/test_results_generated_6_6_seen_cat.txt',
    't5_amr_enriched_results/test_results_amr_enriched_1_5_seen_cat.txt'
]
TEST_RESULTS_FILES_UNSEEN = [
    't5_results/test_results_generated_6_6_unseen_cat.txt',
    't5_amr_enriched_results/test_results_amr_enriched_1_5_unseen_cat.txt'
]
TEST_RESULTS_FILES_SMALL = [
    't5_results/test_results_generated_6_6_small.txt',
    't5_amr_enriched_results/test_results_amr_enriched_1_5_small.txt'
]
TEST_RESULTS_FILES_LARGE = [
    't5_results/test_results_generated_6_6_large.txt',
    't5_amr_enriched_results/test_results_amr_enriched_1_5_large.txt'
]
VALID_RESULT_DEV = [
    't5_results/dev_results_generated_4_3.txt',
    't5_results/dev_results_generated_5_15.txt',
    't5_results/dev_results_generated_5_6.txt',
    't5_results/dev_results_generated_6_6.txt',
]
WEBNLG_RESULT_FILES = [
    'WebNLG_2020_results/Amazon_AI_(Shanghai)/primary.en',
    'WebNLG_2020_results/FBConvAI/primary.en',
    'WebNLG_2020_results/cuni-ufal/primary.en',
    'WebNLG_2020_results/RALI/primary.en',
    'WebNLG_2020_results/Baseline-FORGE2020/primary.en',
]


def get_category_and_triple_count(dataset_split: str):
    dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=dataset_split)
    stats = []
    test_cat_idx = []
    dev_cat_idx = []
    for i, item in enumerate(dataset):
        if i < len(dataset):
            stats.append((item['category'], len(item['input'])))
            if item['category'] in ('MusicalWork', 'Scientist', 'Film'):
                test_cat_idx.append(i)
            else:
                dev_cat_idx.append(i)
        else:
            break
    return stats, test_cat_idx, dev_cat_idx


def read_from_file(filename):
    with open(filename, mode="r", encoding="utf-8") as file:
        result = [line.strip() for line in file]
    return result


def get_reference(data, subset=None):
    if subset is None:
        res = [item['references'] for item in data]
    else:
        subdata = [data[i] for i in subset]
        res = [item['references'] for item in subdata]
    max_len = max(len(sub_array) for sub_array in res)
    res_padded = np.array([sub_array + [''] * (max_len - len(sub_array)) for sub_array in res])
    res_transposed = res_padded.T.tolist()
    return res_transposed


def get_scores(dataset_split: str, result_files_names: list, subset=None):
    """
    Get dataset split parameter (valid or test) and list of files names with generated results.
    Prints BLEU, CHRF2, BLEURT and METEOR scores.
    """
    test_dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=dataset_split)
    references = get_reference(test_dataset, subset)

    results = []
    for file_name in result_files_names:
        results.append(read_from_file(file_name))

    # BLEU and CHRF
    bleu = sacrebleu.metrics.BLEU()
    chrf = sacrebleu.metrics.CHRF(word_order=2)

    # BLEURT
    checkpoint = "bleurt/bleurt-base-128"
    bleurt_scorer = score.BleurtScorer(checkpoint=checkpoint)

    # METEOR
    meteor = load('meteor')

    for i, result in enumerate(results):
        # Compute BLEU and CHRF scores
        bleu_score = bleu.corpus_score(hypotheses=result, references=references)
        chrf_score = chrf.corpus_score(hypotheses=result, references=references)

        # Compute the BLEURT
        bluert_scores_list = []
        for ref in tqdm(references):
            bluert_scores_list.append(bleurt_scorer.score(references=ref, candidates=result, batch_size=200))
        bluert_scores_list = np.array(bluert_scores_list)
        bluert_scores_list = bluert_scores_list.max(axis=0)
        avg_bluert = np.average(bluert_scores_list)

        # Compute the METEOR score
        if subset is None:
            meteor_ref = [item['references'] for item in test_dataset]
        else:
            sub_test_dataset = [test_dataset[i] for i in subset]
            meteor_ref = [item['references'] for item in sub_test_dataset]
        # calculates the average
        meteor_results = meteor.compute(predictions=result, references=meteor_ref)
        meteor_score = meteor_results["meteor"]

        print("\n", result_files_names[i], ">")
        print("BLEU score", round(bleu_score.score, 2))
        print("CHRF score", chrf_score)
        print("BLEURT score", round(avg_bluert, 4))
        print("METEOR score", round(meteor_score, 4))


if __name__ == '__main__':
    get_scores(configs.TEST_SPLIT, TEST_RESULTS_FILES)
