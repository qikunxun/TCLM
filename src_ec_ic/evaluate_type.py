import os
import json
import sys
import torch
import pickle as pkl
from tqdm import tqdm

data_dir = sys.argv[1]
exp = sys.argv[2]
exp_name = sys.argv[3]
raw = False

class Option(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)
            
def load_kg_form_pkl(file_path):
    with open(file_path + '/kg_{}.pkl'.format(target_relation), mode='rb') as fd:
        kg = pkl.load(fd)
    with open(file_path + '/entity2id_{}.pkl'.format(target_relation), mode='rb') as fd:
        entity2id = pkl.load(fd)
    with open(file_path + '/relation2id_{}.pkl'.format(target_relation), mode='rb') as fd:
        relation2id = pkl.load(fd)
    with open(file_path + '/head2id_{}.pkl'.format('ori'), mode='rb') as fd:
        head2id_ori = pkl.load(fd)
    with open(file_path + '/head2id_{}.pkl'.format('inv'), mode='rb') as fd:
        head2id_inv = pkl.load(fd)
    return kg, entity2id, relation2id, head2id_ori, head2id_inv

def load_kg(kg_file, id2entity, id2relation, target_relation=None):
    kg_ori = []
    kg_inv = []
    with open(kg_file, mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            if len(items) != 3: continue
            h, t, r = items
            if id2entity is not None and id2relation is not None:
                h = id2entity[int(h)]
                t = id2entity[int(t)]
                r = id2relation[int(r)]
            if target_relation is not None and r.replace('/', '|') != target_relation: continue
            kg_ori.append((h, r, t))
            kg_inv.append((t, 'INV' + r, h))
    return kg_ori, kg_inv

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

def build_graph(kg):
    graph = {}
    graph_entity = {}
    for triple in kg:
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
    return graph, graph_entity

def extend_graph(graph_entity, valid_data, test_data):
    data = valid_data.copy()
    data.extend(test_data)
    for h, r, t in data:
        if h not in graph_entity:
            graph_entity[h] = {t: [r]}
        else:
            if t in graph_entity[h]:
                graph_entity[h][t].append(r)
            else:
                graph_entity[h][t] = [r]

def init_matrix(matrix, kg, entity2id, entity2id_tail, relation2id):
    print('Processing Matirx(shape={})'.format(matrix.shape))
    for h, r, t in tqdm(kg):
        # if r == target_relation: continue
        if t not in entity2id_tail: continue
        entity_a = entity2id[h]
        entity_b = entity2id_tail[t]
        relation = relation2id[r]
        matrix[entity_a][entity_b][relation] = 1

def get_head(heads, kg):
    entity2id_head = {}
    id2entity_head = {}
    for h in heads:
        entity2id_head[h] = len(entity2id_head)
        id2entity_head[entity2id_head[h]] = h
    for h, r, t in kg:
        if h not in entity2id_head: continue
        if t in entity2id_head: continue
        entity2id_head[t] = len(entity2id_head)
        id2entity_head[entity2id_head[t]] = t
    return entity2id_head, id2entity_head

def extend_by_z(entity2id_head, id2entity_head, kg):
    for h, r, t in kg:
        if h not in entity2id_head: continue
        if t not in entity2id_head:
            entity2id_head[t] = len(entity2id_head)
            id2entity_head[entity2id_head[t]] = t
    return entity2id_head, id2entity_head

def evaluate(id2entity, entity2id, id2relation, relation2id, head2id_ori, head2id_inv, train_kg,
             valid_data, test_data, target_relation):
    print('Entity Num:', len(entity2id))
    print('Relation Num:', len(relation2id))
    print('Train KG Ori Size:', len(train_kg))
    print('Eval KG Ori Size:', len(test_kg))

    graph, graph_entity = build_graph(train_kg)
    extend_graph(graph_entity, valid_data[0], test_data[0])
    extend_graph(graph_entity, valid_data[1], test_data[1])

    total_states_ori = torch.load('{}/{}-{}-{}/state-{}.pt'.format(exp, exp_name, target_relation, 'ori', 'ori'))
    total_states_inv = torch.load('{}/{}-{}-{}/state-{}.pt'.format(exp, exp_name, target_relation, 'ori', 'inv'))
    mrr_tail = 0
    mrr_head = 0
    hit_1_head = 0
    hit_1_tail = 0
    hit_3_head = 0
    hit_3_tail = 0
    hit_10_head = 0
    hit_10_tail = 0
    count_head = 0
    count_tail = 0
    for h, r, t in tqdm(test_data[0]):
        if r.replace('/', '|') != target_relation: continue
        if h not in head2id_ori or t not in entity2id: continue
        truth_score_ori = total_states_ori[head2id_ori[h]][entity2id[t]]
        if raw:
            rank_head = torch.sum((total_states_ori[head2id_ori[h], :] >= truth_score_ori).int())
        else:
            scores_head = total_states_ori[head2id_ori[h], :].clone()
            for tail in graph_entity[h]:
                if tail not in entity2id: continue
                if r in graph_entity[h][tail]: scores_head[entity2id[tail]] = -1e20
            n = torch.sum((scores_head == truth_score_ori).int()) + 1
            m = torch.sum((scores_head > truth_score_ori).int())
            rank_head = m + (n + 1) / 2
        mrr_head += 1 / float(rank_head.item())
        if rank_head <= 1:
            hit_1_head += 1
        else:
            hit_1_head += 0

        if rank_head <= 3:
            hit_3_head += 1
        else:
            hit_3_head += 0

        if rank_head <= 10:
            hit_10_head += 1
        else:
            hit_10_head += 0
        count_head += 1

    for h, r, t in tqdm(test_data[1]):
        if r.replace('/', '|') != 'INV' + target_relation: continue
        if h not in head2id_inv or t not in entity2id: continue
        truth_score_inv = total_states_inv[head2id_inv[h]][entity2id[t]]
        if raw:
            rank_tail = torch.sum((total_states_inv[head2id_inv[h], :] >= truth_score_inv).int())
        else:
            scores_tail = total_states_inv[head2id_inv[h], :].clone()
            for tail in graph_entity[h]:
                if tail not in entity2id: continue
                if r in graph_entity[h][tail]: scores_tail[entity2id[tail]] = -1e20
            n = torch.sum((scores_tail == truth_score_inv).int()) + 1
            m = torch.sum((scores_tail > truth_score_inv).int())
            rank_tail = m + (n + 1) / 2
        mrr_tail += 1 / float(rank_tail.item())
        if rank_tail <= 1:
            hit_1_tail += 1
        else:
            hit_1_tail += 0

        if rank_tail <= 3:
            hit_3_tail += 1
        else:
            hit_3_tail += 0

        if rank_tail <= 10:
            hit_10_tail += 1
        else:
            hit_10_tail += 0
        count_tail += 1

    if count_tail > 0:
        print('mrr tail', mrr_tail)
        mrr_tail /= count_tail
        hit_1_tail /= count_tail
        hit_3_tail /= count_tail
        hit_10_tail /= count_tail
    if count_head > 0:
        mrr_head /= count_head
        hit_1_head /= count_head
        hit_3_head /= count_head
        hit_10_head /= count_head
    print('Target Relation: {}'.format(target_relation))
    print('# of evaluated triples: {}'.format(count_head))
    print('Mrr_head: {}'.format(mrr_head))
    print('Mrr_tail: {}'.format(mrr_tail))
    print('Hit@1_head: {}'.format(hit_1_head))
    print('Hit@1_tail: {}'.format(hit_1_tail))
    print('Hit@3_head: {}'.format(hit_3_head))
    print('Hit@3_tail: {}'.format(hit_3_tail))
    print('Hit@10_head: {}'.format(hit_10_head))
    print('Hit@10_tail: {}'.format(hit_10_tail))
    print('Mrr: {}'.format((mrr_head + mrr_tail) / 2))
    print('Hit@1: {}'.format((hit_1_head + hit_1_tail) / 2))
    print('Hit@3: {}'.format((hit_3_head + hit_3_tail) / 2))
    print('Hit@10: {}'.format((hit_10_head + hit_10_tail) / 2))
    print('=' * 50)
    return mrr_tail, mrr_head, hit_1_head, hit_1_tail, hit_3_head, hit_3_tail, hit_10_head, hit_10_tail, count_head, count_tail

if __name__ == '__main__':
    target_file = 'relation2id.txt'
    total_mrr_tail, total_mrr_head, total_hit_1_head, total_hit_1_tail, total_hit_3_head, total_hit_3_tail, \
    total_hit_10_head, total_hit_10_tail = 0, 0, 0, 0, 0, 0, 0, 0
    total_kg = load_kg('{}/test/Fact.txt'.format(data_dir), None, None, None)
    total_size = len(total_kg[0])
    print(total_size)
    with open('{}/{}'.format(data_dir, target_file), mode='r') as fd:
        for line in tqdm(fd.readlines()):
            if not line: continue
            target_relation = line.strip().split('\t')[0]
            target_relation = target_relation.replace('/', '|')
            try:
                option = Option('{}/{}-{}-{}/'.format(exp, exp_name, target_relation, 'ori'))
                train_kg, entity2id, relation2id, head2id_ori, head2id_inv = load_kg_form_pkl(option.exp_dir)
            except Exception as e:
                print('Failed target relation:', target_relation)
                continue
            id2entity = reverse(entity2id)
            id2relation = reverse(relation2id)
            test_kg = load_kg('{}/test/Fact.txt'.format(option.data_dir), id2entity, id2relation, target_relation)
            valid_kg = ([], [])
            if len(test_kg[0]) == 0 or len(test_kg[1]) == 0: continue
            mrr_tail, mrr_head, hit_1_head, hit_1_tail, hit_3_head, hit_3_tail, \
            hit_10_head, hit_10_tail, count_head, count_tail = evaluate(id2entity, entity2id, id2relation,
                                                       relation2id, head2id_ori, head2id_inv, train_kg, valid_kg,
                                                       test_kg, target_relation)
            total_mrr_tail += mrr_tail * count_tail / total_size
            total_mrr_head += mrr_head * count_head / total_size
            total_hit_1_head += hit_1_head * count_head / total_size
            total_hit_1_tail += hit_1_tail * count_tail / total_size
            total_hit_3_head += hit_3_head * count_head / total_size
            total_hit_3_tail += hit_3_tail * count_tail / total_size
            total_hit_10_head += hit_10_head * count_head / total_size
            total_hit_10_tail += hit_10_tail * count_tail / total_size
    print('Target Relation: ALL')
    print('# of evaluated triples: {}'.format(total_size))
    print('Mrr_head: {}'.format(total_mrr_head))
    print('Mrr_tail: {}'.format(total_mrr_tail))
    print('Hit@1_head: {}'.format(total_hit_1_head))
    print('Hit@1_tail: {}'.format(total_hit_1_tail))
    print('Hit@3_head: {}'.format(total_hit_3_head))
    print('Hit@3_tail: {}'.format(total_hit_3_tail))
    print('Hit@10_head: {}'.format(total_hit_10_head))
    print('Hit@10_tail: {}'.format(total_hit_10_tail))
    print('Mrr: {}'.format((total_mrr_head + total_mrr_tail) / 2))
    print('Hit@1: {}'.format((total_hit_1_head + total_hit_1_tail) / 2))
    print('Hit@3: {}'.format((total_hit_3_head + total_hit_3_tail) / 2))
    print('Hit@10: {}'.format((total_hit_10_head + total_hit_10_tail) / 2))
    print('=' * 50)
