import re
import copy
from treelib import Tree


def read_amr_from_file(from_filename: str):
    """
    Reads AMR structures from a file and organizes them into a list of tuples.
    Args: from_filename (str): Path to the file containing AMR data.
    Returns: list: A list of tuples, each containing (index, target sentence, AMR lines).
    """
    from_file = open(from_filename, "r", encoding='utf-8')
    sentences = from_file.readlines()

    amr_structure = []
    amr_list = []

    len_sentences = len(sentences) - 1
    for i, sent in enumerate(sentences):
        split_colon = sent.split(":")
        if split_colon[0].isdigit():
            idx = split_colon[0]
            split_tok = sent.split("::tok ")
            target = split_tok[1][:-1]
        elif sent == "\n":
            amr_structure.append((idx, target, amr_list))
            amr_list = []
        elif i == len_sentences:
            amr_list.append(sent[:-1])
            amr_structure.append((idx, target, amr_list))
        else:
            amr_list.append(sent[:-1])
    return amr_structure


# calculate the number of failed parsings (containing "rel" tag)
def find_failed_parsing(amr_structure):
    with open("data/failed_parsings.txt", "w", encoding='utf-8') as file:
        failed_count = 0
        for i, item in enumerate(amr_structure):
            idx, target, amr = item
            for line in amr:
                if ':rel' in line:
                    failed_count += 1
                    full_parsing = '\n'.join(amr)
                    file.write("idx %s, %d out of %d:\n%s\n%s\n\n" % (idx, failed_count, i + 1, target, full_parsing))
                    break


def build_amr_tree(amr_structure_item, remove_word_order=True):
    """
    Builds a tree representation of an AMR structure, optionally removing word order annotations.
    Parses an AMR in indented format into a hierarchical tree, while creating a mapping
    from node identifiers to their corresponding concept or name.
    Args:
        amr_structure_item (tuple): A tuple containing (index, target sentence, AMR lines).
        remove_word_order (bool, optional): Whether to remove word order markers (e.g., ~1, ~2) from AMR.
    Returns:
        tuple: A tuple containing the constructed tree and the names dictionary.
    """
    idx, target, amr = amr_structure_item
    # remove leading spaces
    simple_amr = [text.lstrip() for text in amr]
    # remove numbers after tilde
    if remove_word_order:
        simple_amr = [re.sub(r'~\d+', '', text) for text in simple_amr]
    # remove number after PropBank framesets (e.g. want-01, hey-02) BUT leave numbers after spaces (e.g. temperature -1)
    simple_amr = [re.sub(r'(?<=\S)-\d+', '', text) for text in simple_amr]

    def slash_split(line):
        if "/" in line:
            n1, n2 = line.split(" / ", 1)
            names_dict[n1] = n2
            return n1, n2
        else:
            return line, 0

    def single_token_split(line):
        # to catch corner cases where the token after operator is not a short name
        # like ":polarity -", ":mod B", ":mode expressive"
        if (line[0] == '"' or line[0].isdigit() or line[0] == '-' or line.isupper() or len(line) > 2
                or line in {'he', 'it', '.', '+'}):
            return 0, line
        else:
            return line, 0

    # create tree
    tree = Tree()
    prev_node_id = 0
    cur_id = 0
    names_dict = {}  # dictionary for short/long names
    for i, line in enumerate(simple_amr):
        if i == 0:
            line = line.lstrip("(")
            d1, d2 = slash_split(line)
            tree.create_node(tag="root", identifier=0, data=(d1, d2))
            prev_node_id = tree[0].identifier
        else:
            tag, remainder = line.split(" ", 1)
            if remainder[0] == "(" and remainder[-1:] == ")":
                remainder = remainder[1:len(remainder) - 1]
                new_remainder = remainder.rstrip(")")
                step_up = len(remainder) - len(new_remainder)
                d1, d2 = slash_split(new_remainder)

                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
                for _ in range(step_up):
                    prev_node_id = tree.get_node(prev_node_id).predecessor(tree.identifier)
            elif remainder[0] == "(":
                remainder = remainder[1:]
                d1, d2 = slash_split(remainder)
                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
                prev_node_id = tree[cur_id].identifier
            elif remainder[-1:] != ")":
                d1, d2 = single_token_split(remainder)
                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
            else:
                new_remainder = remainder.rstrip(")")
                d1, d2 = single_token_split(new_remainder)
                cur_id += 1
                tree.create_node(parent=prev_node_id, tag=tag, identifier=cur_id, data=(d1, d2))
                step_up = len(remainder) - len(new_remainder)
                for _ in range(step_up):
                    prev_node_id = tree.get_node(prev_node_id).predecessor(tree.identifier)
    return tree, names_dict


def collapse_leaves(tree, names_dict):
    """
    Merges certain leaf nodes into their parents to simplify the tree structure,
    updating the names dictionary accordingly.
    Leaf nodes are collapsed if they are at depth > 2, have a special tag (e.g., ":name"),
    or have a node key of 0. Merged node data combines the parent's and child's names.
    Args:
        tree: The AMR tree structure to simplify.
        names_dict (dict): Dictionary mapping node keys to names.
    Returns:
        tuple: The updated tree and names dictionary after collapsing leaves.
    """
    temp_tree = copy.deepcopy(tree)
    for node in temp_tree.all_nodes_itr():
        if node.is_leaf() and (tree.depth(node.identifier) > 2 or node.data[0] == 0 or node.tag == ":name"):
            parent_id = node.predecessor(tree.identifier)
            parent = tree[parent_id]

            # update the long name
            if node.data[0] != 0:
                node.data = (node.data[0], names_dict[node.data[0]])

            data = '' if parent.data[1] == 'name' else str(parent.data[1]) + ' '

            parent.data = (parent.data[0], data + node.data[1])
            tree.remove_node(node.identifier)
            # update the dict with short/long names
            names_dict[parent.data[0]] = parent.data[1]
    return tree, names_dict


def clean_names_dict(names_dict):
    """
    Cleans and simplifies names in the names dictionary by removing unnecessary words around quoted text or numbers.
    Args: names_dict (dict): A dictionary mapping node keys to their full name strings.
    Returns: dict: The cleaned names dictionary with simplified name values.
    """
    for key in names_dict:
        name = names_dict[key]

        words = name.split()[::-1]
        words_copy = copy.deepcopy(words)

        # remove one word before words in quotes or words before numbers
        # example: govern city "Birmingham" -> govern "Birmingham"
        # example: date-entity 1981 -> 1981
        if len(words) > 1:
            for i, word in enumerate(words):
                if i + 2 < len(words) and word[0] == '"' and words[i + 1][0] != '"' and words[i + 2][0] == '"':
                    words_copy[i + 1] = ','
                elif i + 1 < len(words) and word[0] == '"' and words[i + 1][0] != '"':
                    words_copy[i + 1] = ''
                elif i + 1 < len(words) and word.isdigit() and words[i + 1][0] != '"' and not words[i + 1][0].isdigit():
                    words_copy[i + 1] = ''
        words = words_copy[::-1]

        # remove empty lines
        words = [word for word in words if word]
        # make one line
        words = " ".join(words)
        # remove double quotes
        words_str = words.replace('"', '')

        names_dict[key] = words_str
    return names_dict


def build_one_line_tree(tree):
    """
    Converts a tree structure into a one-line linearized format, preserving hierarchical relationships with parentheses.
    Args: tree: A tree object representing a simplified AMR or similar structure.
    Returns: str: A single-line string representation of the tree.
    """
    one_line = ''
    prev_node_id = 0
    ending_brackets_counter = 0

    # helper function to append node data based on whether it has children or not (embedding)
    def append_node(node, has_children):
        nonlocal one_line, ending_brackets_counter
        tag = re.sub(r'\d+', '', node.tag) if node.tag[:4] == ':ARG' else node.tag
        if has_children:
            one_line += f" {tag} ({node.data[1]}"
            ending_brackets_counter += 1
        else:
            one_line += f" {tag} {node.data[1]}"

    for node in tree.all_nodes_itr():
        if node.tag == 'root':
            one_line = node.data[1]
        else:
            # get node's parent and prev node's parent
            parent_id = node.predecessor(tree.identifier)
            prev_parent_id = tree[prev_node_id].predecessor(tree.identifier)
            has_children = len(tree.children(node.identifier)) > 0

            # if node is under the prev one or on the same level
            if parent_id == prev_node_id or parent_id == prev_parent_id:
                append_node(node, has_children)
            else:
                # backtrack adding closing parentheses for prev nodes
                prev_node_id = tree[prev_node_id].predecessor(tree.identifier)
                while parent_id != prev_node_id:
                    prev_node_id = tree[prev_node_id].predecessor(tree.identifier)
                    one_line += ')'
                    ending_brackets_counter = 0

                append_node(node, has_children)

        prev_node_id = node.identifier

    one_line = one_line + ")" * ending_brackets_counter
    return one_line


def print_tree(tree):
    tree_bytes = tree.show(stdout=False, sorting=False)
    print(tree_bytes.encode('utf-8').decode('utf-8'))


def print_tags(tree):
    for node in tree.all_nodes_itr():
        print(node.tag, node.data)
    print("\n")


# replace phrases in simplified tree with the phrase from surface sentence
# ex: simplified node: :ARG-of 'have-org-role chairman Silvio Berlusconi'
# search for max and min word orders in an initial tree for :ARG-of node (from 8 to 11)
# and replace: :ARG-of 'managed by Sinisa Mihajlovic'
def replace_with_surface_phrase(node, init_tree, target_sent):
    """
    Replaces a simplified node's label with a surface phrase from the original sentence
    based on word order in the initial AMR tree.
    Args:
        node: Node in the simplified tree to update.
        init_tree: Original AMR tree with word order information.
        target_sent (str): Surface sentence corresponding to the AMR.
    Returns:
        bool: True if replacement was successful, False otherwise.
    """
    subtree = init_tree.subtree(node.identifier)
    if '~' not in str(init_tree[node.identifier].data[1]):
        return False
    max_ord = min_ord = int(re.search(r'~(\d+)', init_tree[node.identifier].data[1]).group(1))
    for n in subtree.all_nodes_itr():
        if '~' in str(n.data[1]):
            word_order = int(re.search(r'~(\d+)', n.data[1]).group(1))
            max_ord = max(word_order, max_ord)
            min_ord = min(word_order, min_ord)

    max_min_diff = max_ord - min_ord
    if max_min_diff < 8:
        surface_list = target_sent.split()
        surface_phrase = " ".join(surface_list[min_ord:max_ord + 1])
        node.data = (node.data[0], surface_phrase)
        return True
    else:
        return False


# collapse the leaves and simplify long names
def simplify_tree(amr_structure):
    """
    Simplifies AMR structures by collapsing leaves, shortening long names,
    and building one-line tree representations.
    Args: amr_structure (list): List of (index, target, amr) tuples.
    Returns: list: List of (index, simplified tree) tuples.
    """
    simple_tree_list = []
    for item in amr_structure:
        init_tree, _ = build_amr_tree(item, remove_word_order=False)
        tree, names_dict = build_amr_tree(item, remove_word_order=True)
        idx, target, amr = item

        # add long names where possible
        for node in tree.all_nodes_itr():
            if node.data[1] == 0:
                node.data = (node.data[0], names_dict[node.data[0]])

        for _ in range(3):
            tree, names_dict = collapse_leaves(tree, names_dict)

        # simplify the long names
        names_dict = clean_names_dict(names_dict)
        replaced = set()
        for node in tree.all_nodes_itr():
            key = node.data[0]
            value = names_dict[node.data[0]]

            # update node data by default
            node.data = (key, value)

            if len(value.split()) >= 4 and key not in replaced:
                if replace_with_surface_phrase(node, init_tree, target):
                    names_dict[key] = node.data[1]
                    replaced.add(key)

        simplified_tree = build_one_line_tree(tree)
        simple_tree_list.append((idx, simplified_tree))
    return simple_tree_list


def simplify_tree_save_to_file(amr_structure, filename):
    simple_tree_list = []
    with open(filename, "w", encoding='utf-8') as file:
        for item in amr_structure:
            init_tree, _ = build_amr_tree(item, remove_word_order=False)
            tree, names_dict = build_amr_tree(item, remove_word_order=True)
            idx, target, amr = item

            # add long names where possible
            for node in tree.all_nodes_itr():
                if node.data[1] == 0:
                    node.data = (node.data[0], names_dict[node.data[0]])

            for _ in range(3):
                tree, names_dict = collapse_leaves(tree, names_dict)

            # simplify the long names
            names_dict = clean_names_dict(names_dict)
            replaced = set()
            for node in tree.all_nodes_itr():
                key = node.data[0]
                value = names_dict[node.data[0]]

                # update node data by default
                node.data = (key, value)

                if len(value.split()) >= 4 and key not in replaced:
                    if replace_with_surface_phrase(node, init_tree, target):
                        names_dict[key] = node.data[1]
                        replaced.add(key)

            simplified_tree = build_one_line_tree(tree)
            simple_tree_list.append(simplified_tree)

            file.write("%s\n" % idx)
            file.write("%s\n" % target)
            file.write("%s\n" % simplified_tree)

    return simple_tree_list


# simplifies with detailed comments
def simplify_tree_test(amr_structure):
    for item in amr_structure:
        init_tree, _ = build_amr_tree(item, remove_word_order=False)
        tree, names_dict = build_amr_tree(item, remove_word_order=True)

        idx, target, amr = item

        print('\n', "target sentence:")
        print(target, '\n')

        print("AMR parsing:")
        for a in amr:
            print(a)

        print("initial tree:")
        print_tree(tree)

        # add long names where possible
        for node in tree.all_nodes_itr():
            if node.data[1] == 0:
                node.data = (node.data[0], names_dict[node.data[0]])

        print("initial tags:")
        print_tags(tree)

        for _ in range(3):
            tree, names_dict = collapse_leaves(tree, names_dict)

        print("collapsed tags:")
        print_tags(tree)

        # simplify the long names
        names_dict = clean_names_dict(names_dict)
        replaced = set()
        for node in tree.all_nodes_itr():
            key = node.data[0]
            value = names_dict[node.data[0]]

            # update node data by default
            node.data = (key, value)

            if len(value.split()) >= 4 and key not in replaced:
                if replace_with_surface_phrase(node, init_tree, target):
                    names_dict[key] = node.data[1]
                    replaced.add(key)

        print("simplified tags and tree:")
        print_tags(tree)
        print_tree(tree)

        simplified_amr = build_one_line_tree(tree)
        print("simplified one-line ARM:")
        print(simplified_amr)


def save_to_file(data: list, filename: str):
    with open(filename, "w", encoding='utf-8') as file:
        cur_idx = '1'
        first_line = True

        for idx, item in data:
            wrapped_item = f'[{item}]'

            if first_line:
                file.write(wrapped_item)
                first_line = False
            elif idx == cur_idx:
                file.write(f' {wrapped_item}')
            else:
                file.write(f'\n{wrapped_item}')
            cur_idx = idx


def parse_flat_amr(flat_amr: str):
    """
    Parses a flat AMR structure into a human-readable indented format.
    Args: flat_amr (str): The flat AMR string to parse.
    Returns: str: The parsed AMR with indentation and sentence separation.
    """
    flat_amr_split = flat_amr.split()

    sent_counter = 1
    new_line = False
    tab_number = 0

    parsed_amr = ''
    for token in flat_amr_split:
        if token[0] == '[':
            parsed_amr += 'sentence ' + str(sent_counter) + '\n'
            parsed_amr += token[1:]
            sent_counter += 1
            new_line = True
            tab_number += 1
        elif token.startswith(':') and new_line:
            parsed_amr += '\n' + tab_number*'\t' + token + ' '
            new_line = False
        elif token.startswith(':') and not new_line:
            parsed_amr += '\n' + tab_number*'\t' + token + ' '
        elif token.startswith('('):
            parsed_amr += token[1:]
            new_line = True
            tab_number += 1
        elif token.endswith(')'):
            parsed_amr += token[0:-1]
            new_line = True
            tab_number -= 1
        elif token.endswith(']'):
            new_token = token[0:-1].rstrip(")")
            parsed_amr += new_token + '\n'
            new_line = True
            tab_number = 0
        else:
            parsed_amr += token + ' '
    print(parsed_amr)

    return parsed_amr


def parse_flat_amr_from_file(from_file_name: str, target_file_name: str, subset=None):
    """
    Parses a file with AMR structure into a human-readable indented format.
    Saves targets and readable AMR stuctures to file.
    Args: from_file_name: str: file name with AMR structures,
         target_file_name: str: file name with respective targets.
    """
    subset = subset if subset is not None else None
    with open(from_file_name, "r", encoding='utf-8') as file:
        amr_flat_list = file.readlines()
    with open(target_file_name, "r", encoding='utf-8') as file:
        targets = file.readlines()
    with open(f'{from_file_name[:-4]}_parsed.txt', "w", encoding='utf-8') as file:
        for i, amr in enumerate(amr_flat_list):
            if subset is None:
                amr_parsed = parse_flat_amr(amr)
                file.write(f'{i + 1} {targets[i]}')
                file.write(f'{amr_parsed}\n')
            else:
                if i in subset:
                    amr_parsed = parse_flat_amr(amr)
                    file.write(f'{i+1} {targets[i]}')
                    file.write(f'{amr_parsed}\n')


def linearize_full_amr(from_file: str, to_file: str):
    """
    Linearizes full standard AMRs, without simplification logic applied
    Args: from_file: str: file name with raw AMR structures,
         to_file: str: file name with respective targets.
    """
    amr_structure = read_amr_from_file(from_file)
    linearized_full_amr = []
    #clean AMRs
    prev_idx = '0'
    prev_amr = ''
    for amr_structure_item in amr_structure:
        idx, target, amr = amr_structure_item
        # remove leading spaces
        clean_amr = [text.lstrip() for text in amr]
        # remove numbers after tilde
        clean_amr = [re.sub(r'~\d+', '', text) for text in clean_amr]
        # remove number after PropBank framesets (e.g. want-01, hey-02) BUT leave numbers after spaces (e.g. temperature -1)
        clean_amr = [re.sub(r'(?<=\S)-\d+', '', text) for text in clean_amr]
        # remove double quotes
        clean_amr = [item.replace('"', '') for item in clean_amr]

        amr = " ".join(item for item in clean_amr)
        wrapped_amr = f'[{amr[1:-1]}]'
        if idx == prev_idx:
            linearized_full_amr[int(prev_idx)-1] += ' ' + wrapped_amr
        else:
            linearized_full_amr.append(wrapped_amr)
        prev_idx = idx

    with open(to_file, "w", encoding='utf-8') as file:
        for item in linearized_full_amr:
            file.write("%s\n" % item)


if __name__ == '__main__':
    amr_structure = read_amr_from_file("data/raw_amr_parsing_results/amr_parser_results_test.txt")
    simplified_amr = simplify_tree(amr_structure)
    save_to_file(simplified_amr, "data/amr_parser_simplified_test.txt")

    linearize_full_amr("data/raw_amr_parsing_results/amr_parser_results_train.txt", "data/flat_amr_parsing_results/FULL_amr_parser_results_train_2.txt")
    linearize_full_amr("data/raw_amr_parsing_results/amr_parser_results_test.txt", "data/flat_amr_parsing_results/FULL_amr_parser_results_test_2.txt")
    linearize_full_amr("data/raw_amr_parsing_results/amr_parser_results_valid.txt", "data/flat_amr_parsing_results/FULL_amr_parser_results_validation_2.txt")