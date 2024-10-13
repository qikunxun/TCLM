import os
import numpy as np

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

class Triple:
    def __init__(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t

    def get_triple(self):
        return self.h, self.r, self.t

class Dataset:

    def __init__(self, option=None):
        self.option = option
        self.batch_size = self.option.batch_size
        self.target_relation = self.option.target_relation
        self.kg, self.id2entity, self.entity2id, self.id2relation, self.relation2id, self.train_kg_ori, \
                self.train_kg_inv, self.triple2id, self.id2triple, self.entity_type, \
                self.type2id = self.load_kg_all(self.option.data_dir)
        self.graph, self.graph_entity, self.relation_tail, self.relation_head = self.build_graph()

    def load_kg_all(self, kg_path):
        kg = []
        entities = set()
        relations = set()
        train_path = os.path.join(kg_path, 'train/Fact.txt')
        entity_path = os.path.join(kg_path, 'entity2id.txt')
        relation_path = os.path.join(kg_path, 'relation2id.txt')
        type_path = os.path.join(kg_path, 'type_entity.txt')
        type2id_path = os.path.join(kg_path, 'type2id.txt')

        id2entity, entity2id = {}, {}
        with open(entity_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 2: continue
                entity, idx = items
                entity2id[entity] = int(idx)
                id2entity[int(idx)] = entity

        id2relation, relation2id = {}, {}
        with open(relation_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 2: continue
                relation, idx = items
                relation2id[relation] = int(idx)
                id2relation[int(idx)] = relation

        length = len(relation2id)
        for i in range(length):
            relation2id['INV' + id2relation[i]] = len(relation2id)
            id2relation[len(relation2id) - 1] = 'INV' + id2relation[i]
        relation2id['Identity'] = len(relation2id)
        id2relation[len(id2relation)] = 'Identity'

        train_triples_ori = []
        train_triples_inv = []
        triple2id = {}
        with open(train_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 3: continue
                h, t, r = items
                h = id2entity[int(h)]
                t = id2entity[int(t)]
                r = id2relation[int(r)]
                triple = Triple(h, r, t)
                triple_inv = Triple(t, 'INV' + r, h)
                kg.append(triple)
                kg.append(triple_inv)
                entities.add(h)
                entities.add(t)
                relations.add(r)
                relations.add('INV' + r)
                if r == self.target_relation:
                    train_triples_inv.append(triple_inv)
                    train_triples_ori.append(triple)

                triple_index = '{}\t{}\t{}'.format(h, r, t)
                triple_index_inv = '{}\t{}\t{}'.format(t, 'INV' + r, h)
                if triple_index not in triple2id: triple2id[triple_index] = len(triple2id)
                if triple_index_inv not in triple2id: triple2id[triple_index_inv] = len(triple2id)
        for entity in entity2id:
            triple = Triple(entity, 'Identity', entity)
            triple_index = '{}\t{}\t{}'.format(entity, 'Identity', entity)
            triple2id[triple_index] = -1
            kg.append(triple)
        sorted_id = sorted(triple2id.items(), key=lambda x: x[1])
        for i, item in enumerate(sorted_id):
            triple2id[item[0]] = i
        id2triple = reverse(triple2id)

        entity_type = {}
        with open(type_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) == 1: continue
                for entity in items[1:]:
                    entity_type[int(entity)] = int(items[0])
        type2id = {}
        with open(type2id_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 2: continue
                type, idx = items
                type2id[type] = int(idx)
        return kg, id2entity, entity2id, id2relation, relation2id, train_triples_ori, train_triples_inv, \
                triple2id, id2triple, entity_type, type2id

    def batch_iter(self):
        for i in range(self.option.iteration_per_batch):
            if len(self.train_kg_ori) == 0: break
            batch_ori = np.random.choice(self.train_kg_ori, self.batch_size, replace=True)
            entity_head = {}
            triples = []
            for triple_batch in batch_ori:
                h, r, t = triple_batch.get_triple()
                if h not in entity_head: entity_head[h] = len(entity_head)
                triples.append((h, r, t))
            yield triples, entity_head, False

            entity_head = {}
            triples = []
            batch_inv = np.random.choice(self.train_kg_inv, self.batch_size, replace=True)
            for triple_batch in batch_inv:
                h, r, t = triple_batch.get_triple()
                if h not in entity_head: entity_head[h] = len(entity_head)
                triples.append((h, r, t))
            yield triples, entity_head, True

    def build_graph(self):
        graph = {}
        graph_entity = {}
        relation_head = {}
        relation_tail = {}
        for triple in self.kg:
            h, r, t = triple.get_triple()
            if h not in graph:
                graph[h] = {r: [t]}
                graph_entity[h] = {t: [r]}
            else:
                if r in graph[h]:
                    graph[h][r].append(t)
                else:
                    graph[h][r] = [t]

                if t in graph_entity[h]:
                    graph_entity[h][t].append(r)
                else:
                    graph_entity[h][t] = [r]
            if r == self.target_relation and t not in relation_tail: relation_tail[t] = len(relation_tail)
            if r == self.target_relation and h not in relation_head: relation_head[h] = len(relation_head)
        return graph, graph_entity, relation_tail, relation_head

    def get_targets_tail(self):
        targets = []
        relation_tail = sorted(self.relation_tail.items(), key=lambda d: d[1])
        for item in relation_tail:
            targets.append(item[0])
        return targets

    def get_targets_head(self):
        targets = []
        entities = sorted(self.entity2id.items(), key=lambda d: d[1])
        for item in entities:
            targets.append(item[0])
        return targets

    def select_random_entity_tail(self, targets, cur, h, max_deep=5):
        selected_entity = None
        count = 0
        while selected_entity is None or selected_entity == cur or \
                (h in self.graph and self.target_relation in self.graph[h]
                 and selected_entity in self.graph[h][self.target_relation]):
            if count == max_deep: break
            selected_entity = np.random.choice(targets, 1)[0]
            count += 1
        return selected_entity

    def select_random_entity_head(self, targets, cur, t, max_deep=5):
        selected_entity = None
        count = 0
        while selected_entity is None or selected_entity == cur or \
                (selected_entity in self.graph and self.target_relation in self.graph[selected_entity]
                 and t in self.graph[selected_entity][self.target_relation]):
            # if count == max_deep: break
            selected_entity = np.random.choice(targets, 1)[0]
            count += 1
        return selected_entity

