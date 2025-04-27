from collections import defaultdict
import json
import datasets
import re
import configs


# trivial plan without predicates
def edges_to_bracket(edges):
    """
    Converts a list of edges into a simple bracketed tree structure without including predicates.
    Args: edges (list of tuple): List of (parent, child) pairs representing edges.
    Returns: str: A string representation of the tree in bracketed format.
    Raises: ValueError: If the graph has zero or multiple roots.
    """
    # build a parent -> children map
    children = defaultdict(list)
    nodes = set()
    child_nodes = set()
    for parent, child in edges:
        children[parent].append(child)
        nodes.update([parent, child])
        child_nodes.add(child)

    # find the root (the one node that never appears as a child)
    roots = nodes - child_nodes
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, got: {roots}")
    root = roots.pop()

    # recurse
    def _format(node):
        if node not in children or not children[node]:
            return str(node)
        # format each child (subtree)
        sub = " > ".join(_format(c) for c in children[node])
        return f"[{node} > {sub}]"

    return _format(root)


# trivial plan with predicates
def edges_to_bracket_with_predicates(edges):
    """
    Converts a list of edges with predicates into a bracketed tree structure,
    including predicates.
    """
    # build parent -> list of (child, predicate)
    children = defaultdict(list)
    all_nodes = set()
    child_nodes = set()
    for parent, child, pred in edges:
        children[parent].append((child, pred))
        all_nodes |= {parent, child}
        child_nodes.add(child)

    # find the root
    roots = all_nodes - child_nodes
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, got {roots}")
    root = roots.pop()

    # recursion
    def _format(node):
        # if no children, just print the node
        if node not in children:
            return str(node)

        # otherwise, format each (child, pred) pair
        parts = []
        for child, pred in children[node]:
            # prefix the subtree with its edge‐predicate
            parts.append(f"{pred}:{_format(child)}")
        return f"[{node} > {' > '.join(parts)}]"

    return _format(root)


def save_to_file(data: list, filename: str):
    with open(filename, "w", encoding='utf-8') as file:
        for item in data:
            file.write("%s\n" % item)


def transform_triple(triple):
    """
    Cleans and normalizes a single triple string from WebNLG format,
    Args: triple (str): Triple string formatted with separators (subject|predicate|object).
    Returns: tuple: A tuple (subject, predicate, object) where each component is cleaned and normalized.
    """
    res_triple = triple.replace("_", " ")
    res_triple = res_triple.replace('"', '')
    res_triple = res_triple.replace("'", "")
    triple_split = res_triple.split("|")
    camel_to_space = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', triple_split[1])
    triple_split[1] = camel_to_space.lower()
    if triple_split[2][-2:] == '.0' and triple_split[2][:-2].strip().isdigit():
        triple_split[2] = triple_split[2][:-2]
    return triple_split[0].strip(), triple_split[1].strip(), triple_split[2].strip()


def generate_trivial_plan(json_file, save_file, save_to_file_pred,
                          save_to_file_short, save_to_file_short_pred):
    """
    Generates simple bracketed plans (with and without predicates, full and short versions)
    from a json file containing triples and saves them to specified files.
    Args:
        json_file (str): Path to the input json file containing triples.
        save_file (str): Path to save the full plans without predicates.
        save_to_file_pred (str): Path to save the full plans with predicates.
        save_to_file_short (str): Path to save the short plans without predicates.
        save_to_file_short_pred (str): Path to save the short plans with predicates.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def build(plans, plans_pred, short=False):
        for item in data:
            full_edges, pred_edges = [], []
            for t in item['triples']:
                s, p, o = t['triple']
                if short:
                    s, p, o = (x.split(',', 1)[0] for x in (s, p, o))
                o_label = f"{o} " if s == o else o
                full_edges.append((s, o_label))
                pred_edges.append((s, o_label, p))
            plans.append(edges_to_bracket(full_edges))
            plans_pred.append(edges_to_bracket_with_predicates(pred_edges))

    # build full and short plans, with predicates and without
    full, full_pred = [], []
    short, short_pred = [], []
    build(full, full_pred, short=False)
    build(short, short_pred, short=True)

    save_to_file(full, save_file)
    save_to_file(full_pred, save_to_file_pred)
    save_to_file(short, save_to_file_short)
    save_to_file(short_pred, save_to_file_short_pred)


def generate_trivial_plan_test(save_file, save_to_file_pred,
                               save_to_file_short, save_to_file_short_pred):
    """
    Generates simple bracketed plans (with and without predicates, full and short versions)
    directly from the WebNLG test split and saves them to specified files.
    Args:
        save_file (str): Path to save the full plans without predicates.
        save_to_file_pred (str): Path to save the full plans with predicates.
        save_to_file_short (str): Path to save the short plans without predicates.
        save_to_file_short_pred (str): Path to save the short plans with predicates.
    """
    dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=configs.TEST_SPLIT)

    def build(plans, plans_pred, short=False):
        for item in dataset:
            full_edges, pred_edges = [], []
            for triple in item['input']:
                s, p, o = transform_triple(triple)
                if short:
                    s, p, o = (x.split(',', 1)[0] for x in (s, p, o))
                o_label = f"{o} " if s == o else o
                full_edges.append((s, o_label))
                pred_edges.append((s, o_label, p))
            plans.append(edges_to_bracket(full_edges))
            plans_pred.append(edges_to_bracket_with_predicates(pred_edges))

    # build full and short plans, with predicates and without
    full, full_pred = [], []
    short, short_pred = [], []
    build(full, full_pred, short=False)
    build(short, short_pred, short=True)

    save_to_file(full, save_file)
    save_to_file(full_pred, save_to_file_pred)
    save_to_file(short, save_to_file_short)
    save_to_file(short_pred, save_to_file_short_pred)


def generate_swap_list(json_file, save_to_file, split):
    """
    Generates a swap indicator list for triples in the WebNLG dataset,
    based on annotated swapped edges, and saves the result as json.
    Args:
        json_file (str): Path to the input json file with swap annotations.
        save_to_file (str): Path to save the generated swap dictionary.
        split (str): Dataset split.
    Notes:
        The output json maps each example to a list of booleans indicating whether each triple was swapped.
    """
    with open(json_file, 'r', encoding="utf-8") as file:
        parsed_data = json.load(file)
    dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=split)

    transformed_dataset = []
    swap_dict = {}
    for i, item in enumerate(dataset):
        triples_list = []
        swap_dict[i] = []
        for triple in item['input']:
            s, p, o = transform_triple(triple)
            triples_list.append((s, p, o))
            swap_dict[i].append(False)
        transformed_dataset.append(triples_list)

    for i, parsing in enumerate(parsed_data):
        if len(parsed_data[parsing]["swapped_edges"]) > 0:
            if len(parsed_data[parsing]["swapped_edges"]) > 1:
                bla = 1
            swapped_edges = parsed_data[parsing]["swapped_edges"]
            for swapped_edge in swapped_edges:
                for k, triple in enumerate(transformed_dataset[i]):
                    if (swapped_edge['triple'][0] == triple[2] and swapped_edge['triple'][2] == triple[0] and
                            swapped_edge['triple'][1] == triple[1]):
                        swap_dict[i][k] = True
    with open(save_to_file, "w", encoding="utf-8") as f:
        json.dump(swap_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    generate_trivial_plan(json_file="data/ask_ollama_log/dataset_positions_schema_train_v2.json",
                          save_file="data/flat_plan_results/flat_plan_trivial_train.txt",
                          save_to_file_pred="data/flat_plan_results/flat_plan_trivial_predicates_train.txt",
                          save_to_file_short="data/flat_plan_results/flat_plan_trivial_short_train.txt",
                          save_to_file_short_pred="data/flat_plan_results/flat_plan_trivial_predicates_short_train.txt")
    generate_trivial_plan(json_file="data/ask_ollama_log/dataset_positions_schema_dev.json",
                          save_file="data/flat_plan_results/flat_plan_trivial_validation.txt",
                          save_to_file_pred="data/flat_plan_results/flat_plan_trivial_predicates_validation.txt",
                          save_to_file_short="data/flat_plan_results/flat_plan_trivial_short_validation.txt",
                          save_to_file_short_pred="data/flat_plan_results/flat_plan_trivial_predicates_short_validation.txt")
    generate_trivial_plan_test(save_file="data/flat_plan_results/flat_plan_trivial_test.txt",
                               save_to_file_pred="data/flat_plan_results/flat_plan_trivial_predicates_test.txt",
                               save_to_file_short="data/flat_plan_results/flat_plan_trivial_short_test.txt",
                               save_to_file_short_pred="data/flat_plan_results/flat_plan_trivial_predicates_short_test.txt")

    generate_swap_list('data/plan_schema/flat_plan_train.json', 'data/plan_schema/swap_train.json', configs.TRAIN_SPLIT)
    generate_swap_list('data/plan_schema/flat_plan_dev.json', 'data/plan_schema/swap_dev.json', configs.VALID_SPLIT)
