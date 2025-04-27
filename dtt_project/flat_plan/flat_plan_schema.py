#!/usr/bin/env python3

import datasets
import re
import unicodedata
import fuzzysearch
import ask_ollama.ask_ollama
from tqdm import tqdm
from sentence_splitter import SentenceSplitter
import configs
import json


def remove_diacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def transform_triple(triple):
    """
    Cleans and standardizes a WebNLG triple string.
    Replaces underscores with spaces, removes quotation marks, converts camelCase
    predicates to space-separated lowercase format, and removes unnecessary decimal points from numerical objects.
    Args: triple (str): A triple string in the format 'subject|predicate|object'.
    Returns: tuple: A tuple (subject, predicate, object) with cleaned and standardized text.
    """
    res_triple = triple.replace("_", " ")
    res_triple = res_triple.replace('"', '')
    res_triple = res_triple.replace("'", "")
    triple_split = res_triple.split("|")
    camel_to_space = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', triple_split[1])
    triple_split[1] = camel_to_space.lower()
    if triple_split[2][-2:] == '.0' and triple_split[2][:-2].strip().isdigit():
        triple_split[2] = triple_split[2][:-2]
    res_triple = (triple_split[0].strip(), triple_split[1].strip(), triple_split[2].strip())

    return res_triple


def transform_webnlg_data(dataset, size=None):
    """
    Transforms a WebNLG-style dataset into a structured schema.
    Processes each data item by:
    - Cleaning triples using `transform_triple`.
    - Creating a mapping from subjects and objects to simplified identifiers.
    - Handling cases where subjects or objects share similar prefixes.
    - Preparing a simplified and diacritic-free target text.
    Args:
        dataset (list): A list of dictionaries, each containing 'input' (triples) and 'target' (text).
        size (int, optional): If set, only a subset of the dataset will be processed.
    Returns: list: A list of transformed schemas, each containing triples, a subject-object dictionary, and processed targets.
    """
    # take just 2 first words of subj/object for further search in text
    def first_two_words(phrase):
        phrase_split = phrase.lower().split()
        if len(phrase_split) < 2:
            return remove_diacritics(phrase_split[0])
        # ignore second word: if there is a comma between the words OR the second word is in parentheses OR the
        # second word is "language"
        elif phrase_split[0][-1:] == "," or phrase_split[1][0] == "(" or phrase_split[1] == "language":
            return remove_diacritics(phrase_split[0].replace(',', ''))
        else:
            first_word = remove_diacritics(phrase_split[0])
            second_word = remove_diacritics(phrase_split[1].replace(',', ''))
            return f'{first_word} {second_word}'

    def all_words(phrase):
        return remove_diacritics(phrase.lower().replace(',', ''))

    def create_schema(item, idx):
        subj_obj_dict = {}
        triples = []
        for triple in item['input']:
            s, p, o = transform_triple(triple)
            if s not in subj_obj_dict:
                subj_obj_dict[s] = len(subj_obj_dict) + 1
            if o not in subj_obj_dict:
                subj_obj_dict[o] = len(subj_obj_dict) + 1
            triples.append({"triple": (s, p, o), "schema": (subj_obj_dict[s], subj_obj_dict[o])})

        # sorting by key (string)
        sorted_key_phrase = sorted(subj_obj_dict.keys())

        # simplify key phrases -> first_two_words in case first two words are identical for different key phrases,
        # keep more words in phrases to distinguish them
        subj_obj_dict_simplified = {}
        for i, phrase in enumerate(sorted_key_phrase):
            sub_phrase = first_two_words(phrase)
            if sub_phrase not in subj_obj_dict_simplified:
                subj_obj_dict_simplified[sub_phrase] = subj_obj_dict[phrase]
            else:
                # subj_obj_dict_simplified = {}
                # if prev phrase is of the same length as first_two_words -> keep all words in the current phrase
                # (ex: prev:"addis ababa", cur:"addis ababa hall" -> "addis ababa", "addis ababa hall"
                # if prev phrase is longer than first_two_words -> remove first_two_words from both phrases as it
                # doesn't bring meaning (ex: prev:"addis ababa stadium", cur:"addis ababa hall" -> "stadium", "hall"
                prev_phrase = sorted_key_phrase[i - 1]
                long_sub_phrase = all_words(phrase)
                long_prev_phrase = all_words(prev_phrase)
                if len(prev_phrase) == len(sub_phrase):
                    subj_obj_dict_simplified[long_sub_phrase] = subj_obj_dict[phrase]
                else:
                    # this if-else needed to distinguish more than 3 phrases with identical first 2 words
                    if subj_obj_dict_simplified[sub_phrase] == subj_obj_dict[prev_phrase]:  # we check VALUE here
                        del subj_obj_dict_simplified[sub_phrase]
                        subj_obj_dict_simplified[long_prev_phrase[len(sub_phrase) + 1:]] = subj_obj_dict[prev_phrase]
                    elif long_prev_phrase in subj_obj_dict_simplified:
                        del subj_obj_dict_simplified[long_prev_phrase]
                        subj_obj_dict_simplified[long_prev_phrase[len(sub_phrase) + 1:]] = subj_obj_dict[prev_phrase]
                    subj_obj_dict_simplified[long_sub_phrase[len(sub_phrase) + 1:]] = subj_obj_dict[phrase]

        # unicode removes diacritics
        target = remove_diacritics(item["target"].lower().replace(",", ""))
        schema = {"id": idx+1, "triples": triples, "subj_obj_dict": subj_obj_dict_simplified, "target": target,
                  "initial_target": item["target"]}
        return schema

    transformed = []
    for idx, item in enumerate(dataset):
        if size is not None and not (23610 <= idx < 23610 + size):
            continue
        transformed.append(create_schema(item, idx))
    return transformed


def map_schema_to_target(dataset, annotation=None):
    """
    Maps schema entities from a WebNLG-style dataset to their positions in the target text.
    Attempts to find exact or approximate matches for entity mentions in the target text.
    Uses fuzzy search or LLM assistance if necessary.
    Optionally adjusts mapping using manual annotations if provided.
    Args:
        dataset (list): A list of schemas with triples and targets.
        annotation (dict, optional): A dictionary of manual annotations.
    Returns: list: The input dataset with added 'schema_positions' mapping entity IDs to character positions in the text.
    """
    other_cases = []
    llm_log = []

    for idx, item in enumerate(tqdm(dataset)):
        subj_obj_dict = item["subj_obj_dict"]
        target = item["target"]

        def find_position(k, targ, full_target):
            position = targ.find(k)
            if position != -1:
                return position, position + len(k)

            key_split = k.split()
            # case when key phrase is 1 word
            if len(key_split) == 1:
                position = targ.find(k[0:4])
                # check is needed to avoid cases when key word is part of another word : [targ[position-1] == ' ']
                # (e.g. pop is part of keypop)
                if position != -1 and (targ[position - 1] == ' ' or position == 0):
                    return position, position + 4

            # case when key phrase is 2 words or longer
            if len(key_split) > 1:
                # use fuzzy search only if word is 4-characters or longer
                if len(key_split[0]) > 3:
                    first_pos = fuzzysearch.find_near_matches(subsequence=key_split[0], sequence=targ, max_l_dist=1)
                    first_pos = -1 if len(first_pos) == 0 else first_pos[0].start
                else:
                    first_pos = targ.find(key_split[0])

                if len(key_split[1]) > 3:
                    second_pos = fuzzysearch.find_near_matches(subsequence=key_split[1], sequence=targ, max_l_dist=1)
                    second_pos = -1 if len(second_pos) == 0 else second_pos[0].start
                else:
                    second_pos = targ.find(key_split[1])

                if first_pos != -1 and second_pos != -1 and first_pos < second_pos:
                    # to avoid situations when 1st and 2nd words are far away, return second word position only
                    return second_pos, second_pos + len(key_split[1])
                elif second_pos == -1 and first_pos != -1 and (
                        targ[first_pos - 1] == ' ' or first_pos == 0):  # first word found only
                    return first_pos, first_pos + len(key_split[0])

            # in other cases query LLM
            found_phrase = ask_ollama.ask_ollama.find_phrase(sentence=targ, key=k)
            llm_log.append("id: " + str(
                idx + 1) + "\nkey_split: " + k + "\ntarget: " + targ + "\nfull_target: " + full_target + "\nreply: " + found_phrase + "\n")

            position = targ.find(found_phrase)
            if position != -1:
                print(1)
                return position, position + len(found_phrase)

            # if not found, clean found_phrase and try searching again
            if found_phrase.find("note") != -1:
                found_phrase = found_phrase.split()[0]
            else:
                found_phrase = re.sub(r"[().,]", "", found_phrase)
            if found_phrase[:4].isdigit():
                found_phrase = found_phrase[:4]

            position = targ.find(found_phrase)
            if position != -1:
                print(2)
                return position, position + len(found_phrase)

            if annotation is not None:
                # use the manual annotation file
                if str(idx+1) in annotation.keys() and annotation[str(idx+1)]["key"] == k:
                    print(3)
                    print(annotation[str(idx+1)]["key"], annotation[str(idx+1)]["target"])
                    return annotation[str(idx+1)]["start_position"], annotation[str(idx+1)]["end_position"]

            other_cases.append(
                "id: " + str(
                    idx + 1) + "\nkey_split: " + k + "\ntarget: " + targ + "\nfull_target: " + full_target + "\nreply: " + found_phrase + "\n")

            return None, None

        # Sorting dictionary keys from longest to shortest ->
        # in cases when items with similar beginnings are present {"Aarhus Airport":1, "Aarhus":2}
        # the longest item should be searched first
        sorted_dict = sorted(subj_obj_dict.keys(), key=len, reverse=True)
        positions = {}

        splitter = SentenceSplitter(language='en')
        initial_target_sent = item["initial_target"]
        target_sent_split = splitter.split(initial_target_sent.replace(",", ""))

        # add sentences end positions
        if len(target_sent_split) > 1:
            prev_sent_len = 0
            for i, sent in enumerate(target_sent_split[:-1], start=1):
                prev_sent_len += len(sent)
                positions[prev_sent_len - 1] = f"{i}_sent_end"
                prev_sent_len += 1

        for key in sorted_dict:
            start_position, end_position = find_position(key, target, item["target"])
            # make a pair -> position in target sent : index of phrase
            if start_position is not None:
                positions[int(start_position)] = subj_obj_dict[key]
                # target = target.replace(key, "_" * (end_position - start_position))
                replacement = "_" * (end_position - start_position)
                target = target[:start_position] + replacement + target[end_position:]
            else:
                positions[start_position] = subj_obj_dict[key]
        item["schema_positions"] = positions

    with open("data/ask_ollama_log/DEBUG.txt", "w", encoding='utf-8') as file:
        file.write("\n\nDEBUG_LIST_OTHER\n")
        file.write(str(len(other_cases)) + "\n")
        for item in other_cases:
            file.write("%s\n" % item)

    with open("data/ask_ollama_log/LLM_LOG.txt", "w", encoding='utf-8') as file:
        file.write(str(len(llm_log)) + "\n")
        for item in llm_log:
            file.write("%s\n" % item)

    # save to json
    with open("data/ask_ollama_log/dataset_positions.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    return dataset


# to parse manually annotated file
def parse_annotation(parse_annotation_file):
    """
    Parses a manually annotated file containing phrase positions in target texts.
    Args: parse_annotation_file (str): Path to the manually annotated file.
    Returns: dict: A dictionary mapping data item IDs to their corresponding annotation details.
    """
    from_file = open(parse_annotation_file, "r", encoding="utf-8")
    lines = from_file.readlines()

    items_len = int(lines[0])
    result = {}
    id_idx, key_idx, target_idx, full_target_idx = 1, 2, 3, 4
    for i in range(items_len):
        id = lines[i * 6 + id_idx][4:].strip('\n')
        key = lines[i * 6 + key_idx][11:].strip('\n')
        target = lines[i * 6 + target_idx][8:].strip('\n')
        full_target = lines[i * 6 + full_target_idx][13:].strip('\n')
        if target[0] == "!":
            start_position, end_position = None, None
        else:
            start_position = target.find("*")
            end_position = start_position + target[start_position + 1:].find("*")
        result[id] = {"key": key, "target": target, "full_target": full_target, "start_position": start_position,  "end_position": end_position}
    return result


# adjust schema based on manually annotated file
def adjust_schema_file(json_file_name, annotation_file):
    """
    Adjusts schema positions in a dataset based on manual annotations and saves to json.
    Args:
        json_file_name (str): Path to the json file containing the original schema dataset.
        annotation_file (str): Path to the manual annotation file.
    Returns: None
    """
    annotation = parse_annotation(annotation_file)
    with open(json_file_name, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    annotation = {int(k): v for k, v in annotation.items()}
    for item in loaded_data:
        if item["id"] in annotation:
            search_phrase = annotation[item["id"]]["key"]
            search_phrase_id = item["subj_obj_dict"][search_phrase]
            # removing the item with value search_phrase_id
            item["schema_positions"] = {k: v for k, v in item["schema_positions"].items() if v != search_phrase_id}
            # add adjusted position for search_phrase_id
            item["schema_positions"][annotation[item["id"]]["start_position"]] = search_phrase_id

    with open("data/ask_ollama_log/dataset_positions_dev_after_adj.json", "w", encoding="utf-8") as f:
        json.dump(loaded_data, f, indent=4, ensure_ascii=False)


def create_general_schema(json_file_name):
    """
    Creates a general schema structure from a json file containing processed dataset entries.
    Args: json_file_name (str): Path to the json file with processed schema data.
    Returns: list: A list of general schema entries.
    """
    with open(json_file_name, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    for i, item in enumerate(loaded_data):
        if "null" in item["schema_positions"]:
            item["schema_positions"][-1] = item["schema_positions"].pop("null")
        item["schema_positions"] = {int(k): v for k, v in item["schema_positions"].items()}
        schema_positions_sorted = dict(sorted(item["schema_positions"].items()))

        ordered_pos = []
        sub_ordered_pos = []
        for id in schema_positions_sorted.values():
            # str are ends of sentences only ("1_sent_end")
            if type(id) != str:
                sub_ordered_pos.append(id)
            elif len(item["triples"]) > 1 and len(sub_ordered_pos) > 0:  # ignore sentence ends for items with a single triple
                ordered_pos.append(sub_ordered_pos)
                sub_ordered_pos = []
        if len(sub_ordered_pos) > 0:
            ordered_pos.append(sub_ordered_pos)
        item["general_schema"] = ordered_pos

    with open("data/ask_ollama_log/dataset_positions_schema_dev.json", "w", encoding="utf-8") as f:
        json.dump(loaded_data, f, indent=4, ensure_ascii=False)


def find_positions(annotation_file, split, lang='en'):
    """
    Loads a portion of the WebNLG dataset, applies a transformation,
    and maps annotated schema positions to the corresponding targets.
    Args:
        annotation_file (str): Path to the annotation file containing schema position mappings.
        split (str): Dataset split.
        lang (str, optional): Language version of the WebNLG dataset. Defaults to 'en'.
    Returns: list: A list of transformed dataset entries with added schema position annotations.
    """
    dataset = datasets.load_dataset(path='GEM/web_nlg', name=lang, split=split)
    transformed = transform_webnlg_data(dataset, size=500)

    annotation = parse_annotation(annotation_file)
    dataset_with_positions = map_schema_to_target(transformed, annotation)
    return dataset_with_positions


if __name__ == '__main__':
    find_positions(annotation_file=configs.ANNOTATION_TRAIN, split=configs.TRAIN_SPLIT, lang=configs.LANG)
    adjust_schema_file("data/ask_ollama_log/dataset_positions_dev_before_adj.json", "data/ask_ollama_log/DEBUG_full_dev_new_marked.txt")
    create_general_schema("data/ask_ollama_log/dataset_positions_dev_after_adj.json")

