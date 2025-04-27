import itertools
import re
import json


# class for a node in a tree
class Node:
    def __init__(self, value, is_root=False, children=None, parent=None):
        self.value = value
        self.root = is_root
        self.children = children or []
        self.children_sorted = False
        self.parent = parent

    def __str__(self):
        return f"Node(value={self.value}, children={self.children})"

    def add_child(self, child):
        self.children.append(child)
        self.children_sorted = False

    def order_children(self, order):
        # order children based on the provided order
        sorted_children = sorted(self.children, key=lambda ch: order.get(ch.value))
        for ch in sorted_children:
            ch.order_children(order)

        self.lchildren = [ch for ch in sorted_children if order[ch.value] < order[self.value]]
        self.rchildren = [ch for ch in sorted_children if order[ch.value] > order[self.value]]
        self.children_sorted = True

    def repr(self, incl_pred=False):
        # create a string representation of the node and its children
        if not self.children_sorted:
            print('Children not sorted - possible errors.')

        lout = [f"{ch.repr(incl_pred)} < " for ch in self.lchildren]
        rout = [f" > {ch.repr(incl_pred)}" for ch in self.rchildren]

        pred_id = f"p{''.join(sorted([str(self.parent.value), str(self.value)], key=int))}" if self.parent else ''

        return (
            f"{(not self.root and incl_pred) * f'{pred_id}: '}"
            f"{'[' if not self.root else ''}"
            f"{''.join(lout)}{self.value}{''.join(rout)}"
            f"{']' if not self.root else ''}"
        )

    def cross(self, order, low=None, high=None):
        # check for crossing edges in the tree
        low = low or order[self.value]
        high = high or order[self.value]

        if not self.root and (
                (self.rchildren and max(order[ch.value] for ch in self.rchildren) > high > order[self.value]) or
                (self.lchildren and min(order[ch.value] for ch in self.lchildren) < low < order[self.value])
        ):
            return self.value
        elif not self.children:
            return -1
        else:
            up_low = min(low, order[self.value])
            up_high = max(high, order[self.value])
            return max(ch.cross(order, up_low, up_high) for ch in self.children)

    def list_edges(self):
        return [self.parent.value, self.value] if not self.root else None


#  tree structure built from edges and node order
class TreeStruct:
    def __init__(self,
                 edges: list[tuple[int, int]],
                 order: list[int],
                 parsed: bool = True,
                 id: int = 0):
        self.root_id = None
        self.edges = edges
        self.order_raw = order
        self.id = id
        self.node_dict = {}
        self.parsed = parsed

        self.order = {val: id for id, val in enumerate(order)}
        if parsed:
            self.parse()  # parse tree
            self.sort()  # sort the tree

    def find_root(self):
        id = set([edge[0] for edge in self.edges]) - set([edge[1] for edge in self.edges])
        # assert len(id) == 1, f"Wrong structure, it seems there are 2 roots {self.id}, root {id}"
        self.root_id = int(list(id)[0])
        self.node_dict[self.root_id] = Node(self.root_id, True)

    def parse(self):
        self.find_root()

        for edge in self.edges:
            if edge[1] not in self.node_dict:
                self.node_dict[edge[1]] = Node(edge[1])
            if edge[0] not in self.node_dict:
                self.node_dict[edge[0]] = Node(edge[0], is_root=False, children=[self.node_dict[edge[1]]])
            else:
                self.node_dict[edge[0]].add_child(self.node_dict[edge[1]])

            self.node_dict[edge[1]].parent = self.node_dict[edge[0]]

    def swap_child_with_parent(self, child_id):
        child = self.node_dict[child_id]
        parent = child.parent

        child.parent = parent.parent
        parent.parent = child

        if parent.root:
            parent.root = False
            child.root = True
            self.root_id = child.value

        parent.children.remove(child)
        child.add_child(parent)
        self.sort()

    def sort(self):  # orders all children based on order mapping
        self.node_dict[self.root_id].order_children(self.order)

    def repr(self, trivial=False, incl_pred=False):
        return self.node_dict[self.root_id].repr(incl_pred) if not trivial else " > ".join(map(str, self.order_raw))

    def cross(self):  # check for crossings starting from the root
        return self.node_dict[self.root_id].cross(self.order)

    def get_edges(self):
        return [nd.list_edges() for nd in self.node_dict.values() if
                nd.list_edges() is not None] if self.parse else self.edges


# class for managing tree parsing, checking and shuffling
class TreeHandler:
    def __init__(self,
                 edges: list[tuple[int, int]],
                 order: list[int],
                 parsed: bool = True,
                 id_word_mapping: dict[str, str] = None,
                 allow_shuffling: bool = True,
                 max_iter_shuffle: int = -1,
                 trivial_repr: bool = False,
                 id: int = 0):
        self.edges_original = edges  # internal structure can change -> backup original edges
        self.edges = edges
        self.order = order
        self.parsed = parsed
        self.id_word_mapping = id_word_mapping
        self.id = id
        self.trivial_repr = trivial_repr
        self.max_iter_shuffle = max_iter_shuffle
        self.shuffled = False
        self.nbr_shuffles = 0

        self.parse_status = self.check_tree()

        # deal with non-parsable trees & representation problems
        if self.parse_status[0]:
            self.tree = TreeStruct(edges=edges, order=order, parsed=parsed, id=id)
            if not self.compare_order(trivial=trivial_repr):
                if allow_shuffling:
                    self.shuffle_tree()
                else:
                    self.final_status = False
            else:
                self.final_status = True
        else:
            self.final_status = False
            self.tree = None

    def check_tree(self, incl_non_projectivity=False):
        # check if the tree is valid
        rts = GraphUtils.find_roots(self.edges)
        if len(rts) != 1:
            return False, f'roots: {rts}'

        t = TreeStruct(edges=self.edges, order=self.order, id=self.id)
        id = t.cross()
        flag = id < 0 if incl_non_projectivity else True
        return flag, f"non-projectivity:{id}" if id > 0 else 'ok'

    def shuffle_tree(self, max_iter=None):
        # swap the edges up to max_iter options (by length from the root)
        self.shuffled = True
        vs = GraphUtils.sort_by_dist_to_root(self.edges)

        if max_iter is None:
            max_iter = self.max_iter_shuffle if self.max_iter_shuffle is not None else len(vs)

        if len(vs) > max_iter:
            vs = vs[:max_iter]

        for nbr, v in enumerate(vs):
            edges = GraphUtils.reroot(self.edges, v)
            # create new tree
            tree = TreeStruct(edges=edges, order=self.order, id=self.id)
            status = self.check_tree()
            self.nbr_shuffles = nbr + 1

            if status[0] and self.compare_order(tree=tree):  # found working shuffle
                self.edges = edges
                self.swapped_edges = [e for e in edges if e not in self.edges_original]
                self.tree = tree
                self.final_status = True
                return True

        self.final_status = False
        return False

    def repr(self, trivial: bool = False, incl_pred: bool = False, surface_lvl: bool = False, shorten=False):
        # if tree can't be parsed or there's issue with representation, switch to trivial representation
        if not self.parse_status or not self.final_status:  
            trivial = True
        return self._repr(trivial, incl_pred, surface_lvl, shorten)

    # return symbolic (e.g., 1, 2, 3) or surface representations
    def _repr(self, trivial: bool = False, incl_pred: bool = False, surface_lvl: bool = False,
              shorten: bool = False):
        if trivial or self.trivial_repr:
            base = " > ".join(map(str, self.order))
        else:
            base = self.tree.repr(incl_pred=incl_pred)

        if surface_lvl:
            return self.map_words_to_plan(base, self.id_word_mapping, shorten=shorten)
        else:
            return base

    def compare_order(self, trivial: bool = False, tree: TreeStruct = None):

        if trivial:
            return True

        last_position = -1
        repr = self._repr(trivial) if tree is None else tree.repr(trivial)
        for el in self.order:
            curr_position = repr.find(str(el))
            if curr_position <= last_position:
                last_position = None
                break
            else:
                last_position = curr_position

        return last_position is not None

    def map_words_to_plan(self, plan: str, map_dict: dict[str, str], shorten: bool = False):
        rep = lambda x: map_dict.get(x.group(), x.group()).split(',', 1)[0] if shorten else map_dict.get(x.group(),
                                                                                                         x.group())
        return re.sub(r'\bp\d+|\b\d+\b', rep, plan)  # match either int or pint as a word

    def generate_swapped_edges(self):

        return [{
            'triple': [self.id_word_mapping[str(edge[0])],
                       self.id_word_mapping['p' + ''.join(map(str, sorted(edge)))],
                       self.id_word_mapping[str(edge[1])]
                       ],
            'schema': edge
        }
            for edge in self.swapped_edges
        ] if hasattr(self, 'swapped_edges') else []


# processing utils
class GraphUtils:

    @staticmethod
    def find_roots(edges):
        return set([edge[0] for edge in edges]) - set([edge[1] for edge in edges])

    @staticmethod
    def get_nodes(edges):
        return list(set([v for e in edges for v in e]))

    @staticmethod
    def is_covered(edges, schema):
        return len(set(schema) - set(GraphUtils.get_nodes(edges))) == 0

    @staticmethod
    def drop_edges(edges, schemas):
        # return edges between given nodes
        return [e for e in edges if set(e).issubset(set([v for sch in schemas for v in sch]))]

    @staticmethod
    def sort_by_dist_to_root(edges):
        # sort schema based on distance to root
        root = GraphUtils.find_roots(edges).pop()
        return sorted([v for v in GraphUtils.get_nodes(edges) if v != root], key=lambda x: len(
            GraphUtils.find_path(root, x, edges)))  # tree -> unique path between each two nodes

    @staticmethod
    def find_path(v1, v2, edges):
        end_edge = [e for e in edges if v1 in e and v2 in e]
        if end_edge:
            return [end_edge[0]]
        elif edges:
            for e in [e for e in edges if v1 in e]:
                v1_new = (set(e) - {v1}).pop()
                path = GraphUtils.find_path(v1_new, v2, [ee for ee in edges if e != ee])
                if path is not None:
                    return [e] + path

            return None
        else:
            return None

    @staticmethod
    def reroot(edges, new_root):
        # swap edges so that edges define tree with new root
        root = GraphUtils.find_roots(edges).pop()
        pth = GraphUtils.find_path(root, new_root, edges)

        return [e[::-1] if e in pth else e for e in edges]


class ParseUtils:

    @staticmethod
    def fix_missing_stuff(edges, schemas, missing_nodes):
        mv = missing_nodes
        edgs = edges
        schms = schemas

        while mv:

            schms = [[v for v in sch if v not in mv] for sch in schms]
            edgs = GraphUtils.drop_edges(edgs, schms)

            if not all(GraphUtils.is_covered(edgs, sch) for sch in schms):
                mv = (set(GraphUtils.get_nodes(edgs)) ^ set([v for sch in schms for v in sch]))
            else:
                mv = None
        return edgs, schms

    @staticmethod
    def extend_schemas(edges, schemas):
        # given schemas and edges, extend schemas in order for them to be covered by edges
        unused_edges = [e for e in edges if all(not set(e).issubset(sch) for sch in schemas)]

        # generate fixes for all schemas which are not covered by edges and then find a solution which uses all the
        # edges at most once
        broken_schemas = {o: sch for o, sch in enumerate(schemas) if
                          not GraphUtils.is_covered([e for e in edges if set(e).issubset(sch)], sch)}
        correct_schemas = {o: sch for o, sch in enumerate(schemas) if
                           GraphUtils.is_covered([e for e in edges if set(e).issubset(sch)], sch)}

        fix_space = {
            o: ParseUtils.fix_schema([e for e in edges if set(e).issubset(sch)],
                                     unused_edges,
                                     sch)
            for o, sch in broken_schemas.items() if
            len(ParseUtils.fix_schema([e for e in edges if set(e).issubset(sch)],
                                      unused_edges,
                                      sch)) > 0
        }

        # loop through solutions and return updated_schemas if solution is found
        for combination in itertools.product(*fix_space.values()):
            combo_dict = dict(zip(fix_space.keys(), combination))
            used_edges = [tuple(e) for itm in combination for e in itm]
            if len(used_edges) == len(set(used_edges)):  # check if all edges are used only once
                # update schemas and return them
                upd_schemas = {
                    o: (list(set(GraphUtils.get_nodes(combo_dict[o])) - set(broken_schemas[o])) + broken_schemas[
                        o]) if o in combo_dict.keys() else broken_schemas[o]
                    for o in broken_schemas.keys()
                }
                return [upd_schemas[o] if o in upd_schemas.keys() else correct_schemas[o] for o in range(len(schemas))]
        return None

    @staticmethod
    def fix_schema(current_edges, unused_edges, schema):
        # For given schema find minimal sets of edges which after adding to the schema (together with missing
        # nodes) make the connected graph, finds them based on the set of unused edges
        if len(schema) == 1:
            return [[e] for e in unused_edges if schema[0] in e]

        possibilities = []
        rel_edges = [e for e in unused_edges if e[1] in schema or e[0] in schema]

        for l in range(1, len(rel_edges) + 1):
            opts = itertools.combinations(rel_edges, l)
            for opt in opts:
                if GraphUtils.is_covered(current_edges + list(opt), schema):
                    possibilities.append(list(opt))
        return possibilities

    @staticmethod
    def fix_case(edges, schemas):
        new_schemas = ParseUtils.extend_schemas(edges, schemas)

        if new_schemas:
            return new_schemas

        trivial_schemas = [sch for sch in schemas if len(sch) == 1]
        for orphan in trivial_schemas:
            el = orphan[0]
            rel_nodes = set([e[0] if el == e[1] else e[1] for e in edges if el in e])
            for acceptor in [sch for sch in schemas if sch != orphan and len(rel_nodes.intersection(sch))]:
                new_schemas = ParseUtils.extend_schemas(edges, [sch if sch != acceptor else orphan + acceptor for sch in
                                                                schemas if sch != orphan])
                if new_schemas:
                    return new_schemas

        return None

    @staticmethod
    def clean_up_data(data):
        # fix impossible combinations of schemas and triples, checks for items of triples which
        # are missing and checks for triples of type (x,x)
        new_data = {}
        for id, itm in data.items():
            schemas = itm['schemas']
            edges = itm['edges']

            if itm['errors']['missing_pieces'] > 0 or itm['errors']['not_found']:
                missing_nodes = set(i for e in itm['edges'] for i in e) - set(
                    i for sch in itm['schemas'] for i in sch) | itm['errors']['missing_nodes']

                edges, schemas = ParseUtils.fix_missing_stuff(edges, schemas, missing_nodes)
                # if id in (35007,):
                #     print(f"{id=} {missing_nodes=} {edges=}, {schemas=}")

            if itm['errors']['trivial_edges'] > 0:
                edges = [e for e in edges if e[0] != e[1]]
                if len(edges) == 0:
                    continue

            if itm['errors']['trivial_schemas'] > 0 or itm['errors']['uncovered_nodes'] > 0:
                new_schemas = ParseUtils.fix_case(edges, schemas)
                # if id == 21279:
                #     print('old', schemas, 'new', new_schemas)
                if new_schemas:
                    schemas = new_schemas

            new_data[id] = {'errors': {}, }
            new_data[id]['schemas'] = schemas
            new_data[id]['edges'] = edges

        return new_data

    # quality checks
    @staticmethod
    def detect_incorrect_data(parsed_data):
        for id, itm in parsed_data.items():
            missing_pieces = set(i for e in itm['edges'] for i in e) - set(i for sch in itm['schemas'] for i in sch)

            unc = False
            for sch in itm['schemas']:
                covered_nodes = GraphUtils.get_nodes([e for e in itm['edges'] if set(e).issubset(sch)])
                if len(set(sch) - set(covered_nodes)) > 0:
                    unc = True

            parsed_data[id]['errors']['missing_pieces'] = len(missing_pieces)
            parsed_data[id]['errors']['trivial_edges'] = len([e for e in itm['edges'] if e[0] == e[1]])
            parsed_data[id]['errors']['trivial_schemas'] = len([sch for sch in itm['schemas'] if len(sch) == 1])
            parsed_data[id]['errors']['uncovered_nodes'] = unc

        return parsed_data


# post parse utils
def fix_cross_tree(tree, max_iter=10):
    counter = 0
    while counter < max_iter and tree.cross() > 0:
        tree.swap_child_with_parent(tree.cross())
        counter += 1

    return tree


# create a dataset of triples and general schemas
def parse_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def get_id_word_map(case_data):
        word_id_map = {}
        for triple_entry in case_data['triples']:
            schema = triple_entry['schema']
            triple = triple_entry['triple']

            # nodes
            subj_id, obj_id = str(schema[0]), str(schema[1])
            subj_word, obj_word = triple[0], triple[2]
            word_id_map.setdefault(subj_id, subj_word)
            word_id_map.setdefault(obj_id, obj_word)

            # predicate
            pred_key = 'p' + ''.join(map(str, sorted(schema)))
            word_id_map.setdefault(pred_key, triple[1])

        return word_id_map

    parsed_cases = {}

    for case in data:
        case_id = case['id']
        triples = case['triples']
        schema_positions = case['schema_positions']

        # edges
        edges = [triple['schema'] for triple in triples]

        # find errors
        not_found_error = "-1" in schema_positions
        missing_nodes = {item for id_key, item in schema_positions.items() if "-" in id_key}
        errors = {
            'not_found': not_found_error,
            'missing_nodes': missing_nodes
        }

        schemas = case['general_schema']
        id_word_mapping = get_id_word_map(case)
        target_text = case['target']
        parsed_cases[case_id] = {
            'edges': edges,
            'errors': errors,
            'schemas': schemas,
            'id_word_mapping': id_word_mapping,
            'target': target_text
        }

    return parsed_cases


if __name__ == '__main__':
    parsed_data = parse_json("data/ask_ollama_log/dataset_positions_schema_train_v4.json")
    # check for incorrect cases
    parsed_data = ParseUtils.detect_incorrect_data(parsed_data)
    cleaned_data = ParseUtils.clean_up_data(parsed_data)
    cleaned_data = ParseUtils.detect_incorrect_data(cleaned_data)
    cases = {
        id: [TreeHandler(edges=[i for i in ex['edges'] if set(i).issubset(sch)],
                         order=sch,
                         id_word_mapping=parsed_data[id]['id_word_mapping'],
                         id=id)

             for sch in ex['schemas']

             ]
        for id, ex in cleaned_data.items()}
    representations = {
        id: {'representations':
                 {'trivial_symbolic': [tr.repr(trivial=True) for tr in itm],
                  'trivial_surface': [tr.repr(trivial=True, surface_lvl=True) for tr in itm],
                  'symbolic': [tr.repr() for tr in itm],
                  'symbolic_p': [tr.repr(incl_pred=True) for tr in itm],
                  'surface': [tr.repr(surface_lvl=True) for tr in itm],
                  'surface_p': [tr.repr(surface_lvl=True, incl_pred=True) for tr in itm],
                  'surface_short': [tr.repr(surface_lvl=True, shorten=True) for tr in itm],
                  'surface_p_short': [tr.repr(surface_lvl=True, incl_pred=True, shorten=True) for tr in itm]},
             'swapped_edges': [e for tr in itm for e in tr.generate_swapped_edges()]
             }
        for id, itm in cases.items()
    }

    with open("flat_plan_dev.json", "w") as f:
        json.dump(representations, f, indent=2)
