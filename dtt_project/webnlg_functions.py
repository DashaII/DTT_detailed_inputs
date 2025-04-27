import datasets
import configs
import os


def dataset_stats(split):
    data_en = datasets.load_dataset('GEM/web_nlg', 'en', split=split)

    category_counter = {}
    triples_counter = {}
    for item in data_en:
        triple_count = len(item["input"])
        item_cat = item["category"]
        category_counter[item_cat] = category_counter.get(item_cat, 0) + 1
        triples_counter[triple_count] = triples_counter.get(triple_count, 0) + 1

    category_counter = dict(sorted(category_counter.items()))
    triples_counter = dict(sorted(triples_counter.items()))

    print(split, len(data_en))
    print(category_counter)
    print(triples_counter)


def print_data_to_file(split: str, filename: str):
    data_en = datasets.load_dataset('GEM/web_nlg', 'en', split=split)
    with open(filename, "w", encoding='utf-8') as file:
        for item in data_en:
            # file.write("%s\n" % str(item['input']))
            file.write("%s\n" % item['target'])


def split_seen_and_unseen_categories(split, from_file_name):
    """
    Splits the input lines into 'seen' and 'unseen' categories
    based on the WebNLG dataset, and saves them to new files.
    Args:
        split (str): Dataset split.
        from_file_name (str): Path to the file with lines to split.
    Returns: tuple: Two lists – seen lines and unseen lines.
    """
    unseen_categories = {"Film", "MusicalWork", "Scientist"}
    data_en = datasets.load_dataset('GEM/web_nlg', 'en', split=split)

    seen_idx = [i for i, item in enumerate(data_en) if item["category"] not in unseen_categories]
    unseen_idx = [i for i, item in enumerate(data_en) if item["category"] in unseen_categories]

    with open(from_file_name, "r", encoding='utf-8') as file:
        lines = file.readlines()

    seen_lines = [lines[i] for i in seen_idx]
    unseen_lines = [lines[i] for i in unseen_idx]

    # create new file paths
    base, ext = os.path.splitext(from_file_name)
    seen_file = f"{base}_seen_cat{ext}"
    unseen_file = f"{base}_unseen_cat{ext}"

    with open(seen_file, "w", encoding='utf-8') as f:
        f.writelines(seen_lines)
    with open(unseen_file, "w", encoding='utf-8') as f:
        f.writelines(unseen_lines)

    return seen_lines, unseen_lines


def split_triples_length(split, from_file_name):
    """
    Splits the input lines into two groups based on the number of triples.
    Lines corresponding to items with 3 or fewer triples are saved as "small",
    while those with 4 or more triples are saved as "large".
    Args:
        split (str): Dataset split.
        from_file_name (str): Path to the file with lines to split.
    Returns:
        tuple: Two lists – small triple lines and large triple lines.
    """
    data_en = datasets.load_dataset('GEM/web_nlg', 'en', split=split)

    small_triples_idx = [i for i, item in enumerate(data_en) if len(item["input"]) <= 3]
    large_triples_idx = [i for i, item in enumerate(data_en) if len(item["input"]) >= 4]

    with open(from_file_name, "r", encoding='utf-8') as file:
        lines = file.readlines()

    small_lines = [lines[i] for i in small_triples_idx]
    large_lines = [lines[i] for i in large_triples_idx]

    base, ext = os.path.splitext(from_file_name)
    small_file = f"{base}_small{ext}"
    large_file = f"{base}_large{ext}"

    with open(small_file, "w", encoding='utf-8') as f:
        f.writelines(small_lines)
    with open(large_file, "w", encoding='utf-8') as f:
        f.writelines(large_lines)

    return small_lines, large_lines


if __name__ == '__main__':
    dataset_stats(configs.TRAIN_SPLIT)
    dataset_stats(configs.VALID_SPLIT)
    dataset_stats(configs.TEST_SPLIT)

    seen, unseen = split_seen_and_unseen_categories(configs.TEST_SPLIT, "t5_results/test_results_generated_6_6.txt")
    small, large = split_triples_length(configs.TEST_SPLIT, "t5_amr_enriched_results/test_results_amr_enriched_1_5.txt")

