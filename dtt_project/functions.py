import configs
import datasets
import random
import ast


def save_to_file(data1: list, data2: list, filename: str):
    """
    Merges 2 datasets.
    """
    with open(filename, "w", encoding='utf-8') as file:
        for i, (i1, i2) in enumerate(zip(data1, data2)):
            if not i % 2:
                file.write("%s" % i1)
            else:
                file.write("%s" % i2)


def pick_random_examples(split, number=50, min_triples_num=3):
    """
    Picks random examples from a dataset based on input conditions.
    Parameters:
        split (str): Dataset split to load (e.g., "train", "validation").
        number (int): Number of examples to pick.
        min_triples_num (int): Minimum number of triples required in an example.
        random_seed (int, optional): Random seed for reproducibility.
    Returns:
        set: Indices of picked examples.
    """
    dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=split)
    dataset_length = len(dataset)

    picked_idx = set()
    picked_triple_starts = set()
    counters = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    slice_length = 10
    while len(picked_idx) != number:
        idx = random.randint(0, dataset_length-1)
        if idx in picked_idx:
            continue

        triples = dataset[idx]['input']

        if len(triples) < min_triples_num:  # if triples # < minimum allowed triples #
            continue
        if len(triples) == min_triples_num and idx % 3:  # decrease number of min triples number examples
            continue
        if triples[0][:slice_length] in picked_triple_starts:
            continue
        picked_idx.add(idx)
        picked_triple_starts.add(dataset[idx]['input'][0][:slice_length])

        triples = dataset[idx]['input']
        if len(triples) in counters:
            counters[len(triples)] += 1

    for num, count in counters.items():
        print(f"{num}: {count / number:.2f}")
    return picked_idx


def split_btw_two_groups(length, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    indices = list(range(length))
    random.shuffle(indices)

    list1 = sorted(indices[:int(length/2)])
    list2 = sorted(indices[int(length/2):])
    return list1, list2


def create_subset_for_analysis(full_indices, split_idx1, dataset_split, file1, file2):
    """
    Creates two subsets of data for error analysis by splitting and reorganizing model outputs,
    and saves the results alongside the corresponding triples.
    Args:
        full_indices (list or None): List of indices to extract a subset from the full dataset, or None if data files are already filtered.
        split_idx1 (list): List of indices specifying which examples belong to the first group.
        dataset_split (str): Dataset split.
        file1 (str): Path to the first file containing outputs.
        file2 (str): Path to the second file containing outputs.
    Notes:
        - If `full_indices` is provided, the function extracts a subset from the full dataset and splits it according to `split_idx1`.
        - If `full_indices` is None, the function assumes that `file1` and `file2` already contain only the evaluation examples.
    """
    def save_result(filename, subset_tr, result):
        with open(filename, "w", encoding='utf-8') as file:
            for (t, r) in zip(subset_tr, result):
                for triple in t:
                    file.write("%s\n" % triple)
                file.write("%s\n" % r)

    with open(file1, "r", encoding='utf-8') as file:
        d1 = file.readlines()
    with open(file2, "r", encoding='utf-8') as file:
        d2 = file.readlines()

    # if data files contain full set (like validation) and subset is needed
    if full_indices is not None:
        dataset1 = [d1[i] for i in full_indices]
        dataset2 = [d2[i] for i in full_indices]

        result_left = []
        result_right = []
        for idx, value in enumerate(full_indices):
            if idx in split_idx1:
                result_left.append(dataset1[idx])
                result_right.append(dataset2[idx])
            else:
                result_left.append(dataset2[idx])
                result_right.append(dataset1[idx])

        dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=dataset_split)
        subset_triples = [dataset[i]['input'] for i in full_indices]
    else:  # if data files contain just data for evaluation
        result_left = []
        result_right = []
        for idx in range(len(d1)):
            if idx in split_idx1:
                result_left.append(d1[idx])
                result_right.append(d2[idx])
            else:
                result_left.append(d1[idx])
                result_right.append(d2[idx])

        subset_triples = parse_plain_triples_file(configs.CUSTOM_PATH)

    save_result("data/error_analysis/dev_error_analysis_2a.txt", subset_triples, result_left)
    save_result("data/error_analysis/dev_error_analysis_2b.txt", subset_triples, result_right)


def parse_plain_triples_file(filename):
    # read triples from file
    with open(configs.CUSTOM_PATH, "r", encoding='utf-8') as file:
        dataset = file.readlines()
    return [ast.literal_eval(line.strip()) for line in dataset]


if __name__ == '__main__':
    create_subset_for_analysis(None, configs.IDX_1, configs.CUSTOM_SPLIT,
                               file1="t5_results/baseline_results_unique50_1_15.txt",
                               file2="t5_amr_enriched_results/results_amr_en_unique50_5_6.txt")

