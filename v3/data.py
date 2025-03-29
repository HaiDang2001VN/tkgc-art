# updated data.py
import gzip
import numpy as np
import torch
from torch_geometric.data import Data
from easydict import EasyDict as edict


class BaseDataset:
    """
    BaseDataset is a unified class for loading datasets.
    It loads entities, reviews, and relation files from gzipped text files.
    Reviews are expected to have two tab-separated fields (e.g. user and main entity).
    Relation files are expected to align with the main entity vocabulary.
    """

    def __init__(self, data_dir, set_name='train', entity_files=None, review_file=None, relation_files=None):
        self.data_dir = data_dir if data_dir.endswith('/') else data_dir + '/'
        self.set_name = set_name
        self.review_file = review_file if review_file is not None else f'{set_name}.txt.gz'
        self.entity_files = entity_files if entity_files is not None else {}
        self.relation_files = relation_files if relation_files is not None else {}

        self.load_entities()
        self.load_relations()
        self.load_reviews()

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'rt') as f:
            return [line.strip() for line in f]

    def load_entities(self):
        # Record entity types in the order given by the mapping.
        self.entity_order = list(self.entity_files.keys())
        for entity_name, filename in self.entity_files.items():
            vocab = self._load_file(filename)
            setattr(self, entity_name, edict(
                vocab=vocab, vocab_size=len(vocab)))
            print(f'Loaded entity {entity_name} with size {len(vocab)}.')

    def load_reviews(self):
        # Load review data as pairs. Assume that the second entity is the “main” entity.
        review_data = []
        entity_keys = list(self.entity_files.keys())
        self.main_entity = entity_keys[1] if len(entity_keys) >= 2 else None
        if self.main_entity is None:
            raise ValueError("At least two entity types are required.")
        for line in self._load_file(self.review_file):
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            user_idx, main_idx = int(parts[0]), int(parts[1])
            review_data.append((user_idx, main_idx))
        self.review = edict(data=review_data, size=len(review_data))
        print(f'Loaded {len(review_data)} reviews.')

    def load_relations(self):
        # Load relation files; each relation file is assumed to align with the main entity vocab.
        for relation_name, (filename, target_entity) in self.relation_files.items():
            target_entity_obj = getattr(self, target_entity, None)
            if target_entity_obj is None:
                raise ValueError(
                    f"Target entity '{target_entity}' for relation '{relation_name}' not loaded.")
            vocab_size = target_entity_obj.vocab_size
            et_distrib = np.zeros(vocab_size)
            relation_data = []
            for line in self._load_file(filename):
                items = [int(x) for x in line.split() if x]
                for x in items:
                    et_distrib[x] += 1
                relation_data.append(items)
            setattr(self, relation_name, edict(data=relation_data,
                                               et_vocab=target_entity_obj.vocab,
                                               et_distrib=et_distrib))
            print(
                f'Loaded relation {relation_name} with {len(relation_data)} entries.')


class KGDataset(BaseDataset):
    """
    KGDataset extends BaseDataset to build a unified triple set and converts it into a 
    PyTorch Geometric Data object. It creates a contiguous global index space for all entities.
    Triples are built from both reviews and relation files.
    """

    def __init__(self, data_dir, set_name='train', entity_files=None, review_file=None, relation_files=None):
        super(KGDataset, self).__init__(data_dir, set_name,
                                        entity_files, review_file, relation_files)
        self.build_entity_mappings()
        self.triples = self.build_triples()

    def build_entity_mappings(self):
        """
        Build mapping from (entity_type, local_id) to global id.
        Global ids are assigned consecutively according to the order in self.entity_order.
        """
        self.entity_offsets = {}
        offset = 0
        for etype in self.entity_order:
            vocab_size = getattr(self, etype).vocab_size
            self.entity_offsets[etype] = offset
            offset += vocab_size
        self.num_entities = offset
        print(f'Total number of entities: {self.num_entities}')

    def build_triples(self):
        """
        Build triples as a list of (head, relation, tail) tuples with global ids.
        Reviews are converted to triples using a default review relation:
            - 'purchase' if main entity is 'product'
            - 'view' if main entity is 'recipe'
            - otherwise, 'review'
        Then, for each relation file, triples are built from the main entity to the target entity.
        """
        triples = []
        # Determine the review relation name.
        if self.main_entity == 'product':
            review_relation = 'purchase'
        elif self.main_entity == 'recipe':
            review_relation = 'view'
        else:
            review_relation = 'review'
        # Create a relation mapping: include the review relation plus those from relation_files.
        relation_names = [review_relation] + list(self.relation_files.keys())
        self.relation2id = {rel: idx for idx, rel in enumerate(relation_names)}
        self.num_relations = len(self.relation2id)
        print(f'Relation mapping: {self.relation2id}')

        # Build review triples: (user, review_relation, main_entity)
        for (user_idx, main_idx) in self.review.data:
            head = self.entity_offsets['user'] + user_idx
            tail = self.entity_offsets[self.main_entity] + main_idx
            triples.append((head, self.relation2id[review_relation], tail))

        # Build relation triples from relation files.
        for rel_name in self.relation_files:
            target_entity = self.relation_files[rel_name][1]
            rel_data = getattr(self, rel_name).data
            # Each line in the relation file corresponds to a main entity index.
            for i, tail_list in enumerate(rel_data):
                head = self.entity_offsets[self.main_entity] + i
                for local_tail in tail_list:
                    tail = self.entity_offsets[target_entity] + local_tail
                    triples.append((head, self.relation2id[rel_name], tail))
        triples = torch.tensor(triples, dtype=torch.long)
        print(f'Built {triples.size(0)} triples.')
        return triples

    def get_pyg_data(self):
        """
        Convert the triples into a PyTorch Geometric Data object.
        Attributes:
            edge_index: LongTensor of shape [2, num_triples] (head and tail indices)
            edge_type: LongTensor of relation type ids for each triple
            num_nodes: Total number of entities in the graph
        """
        edge_index = self.triples[:, [0, 2]].t(
        ).contiguous()  # shape [2, num_triples]
        edge_type = self.triples[:, 1]
        data = Data(edge_index=edge_index, edge_type=edge_type,
                    num_nodes=self.num_entities)
        return data
