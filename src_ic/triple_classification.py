import os
import json
import torch
import pickle as pkl
from model import TCLMModel
from sklearn import metrics
from main import create_graph
batch_size = 64
use_gpu = False


class Option(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)

def load_kg_form_pkl(file_path, target_relation):
    with open(file_path + 'kg_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        kg = pkl.load(fd)
    with open(file_path + 'entity2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        entity2id = pkl.load(fd)
    with open(file_path + 'relation2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        relation2id = pkl.load(fd)
    with open(file_path + 'triple2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        triple2id = pkl.load(fd)
    return kg, entity2id, relation2id, triple2id

def load_kg(kg_file):
    kg = []
    with open(kg_file, mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            if len(items) != 3: continue
            h, r, t = items
            kg.append((h, r, t))
    return kg

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

def build_graph(kg, target_relation):
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


def evaluate(relation, entity2id, triple2id, relation2id, train_kg, eval_kg, option, model_save_path):
    print('Entity Num:', len(entity2id))
    print('Relation Num:', len(relation2id))
    print('Train KG Size:', len(train_kg))
    print('Eval KG Size:', len(eval_kg))
    model = TCLMModel(len(relation2id), option.step, option.length,
                      len(entity2id), option.tau_1, option.tau_2, use_gpu)
    print(model_save_path)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    if use_gpu: model = model.cuda()
    model.eval()
    e2triple, triple2e, triple2r = create_graph(train_kg, entity2id, relation2id, triple2id, use_gpu)
    id2head = {}
    head2id = {}
    for h, t, label in eval_kg:
        if h not in head2id:
            head2id[h] = len(head2id)
            id2head[head2id[h]] = h
    print(option.target_relation, len(head2id))
    if len(head2id) == 0: exit(0)
    if len(head2id) % batch_size == 0:
        batch_num = int(len(head2id) / batch_size)
    else:
        batch_num = int(len(head2id) / batch_size) + 1

    total_states = []
    for i in range(batch_num):
        cur_szie = batch_size
        if i == batch_num - 1:
            cur_szie = len(head2id) - i * batch_size

        entity2id_head = {}
        for j in range(cur_szie):
            entity2id_head[id2head[i * batch_size + j]] = j
        input_x = []
        for x in entity2id_head.keys():
            x = entity2id[x]
            input_x.append(x)
        input_x = torch.LongTensor(input_x)
        input_x = torch.nn.functional.one_hot(input_x, len(entity2id)).float().to_sparse()
        if use_gpu: input_x = input_x.cuda()
        state = model(input_x, e2triple, triple2e, triple2r, False, is_training=False)
        total_states.append(state.cpu().detach())
    total_states = torch.cat(total_states, dim=0)
    results = []
    for h, t, label in eval_kg:
        if h not in head2id or t not in entity2id: continue
        truth_score = total_states[head2id[h]][entity2id[t]]
        results.append((h, relation, t, label, truth_score.item()))
        if label != '0': print(h, relation, t, label, truth_score.item())
    return results

def search_thr(results):
    thrs = range(-500, 500)
    max_acc = 0
    best_thr = 0
    for thr in thrs:
        thr = thr * 0.1
        y_pred = []
        y_true = []
        for h, relation, t, label, score in results:
            pred = 0
            if score > thr: pred = 1
            y_pred.append(pred)
            y_true.append(int(label))
        acc = metrics.accuracy_score(y_true, y_pred)
        if acc > max_acc:
            max_acc = acc
            best_thr = thr
    return max_acc, best_thr

def apply_thr(results, thr):
    y_pred = []
    y_true = []
    for h, relation, t, label, score in results:
        pred = 0
        if score > thr: pred = 1
        y_pred.append(pred)
        y_true.append(int(label))
    return y_true, y_pred

if __name__ == '__main__':
    valid_data = {}
    test_data = {}
    dataset = 'umls'
    triples = set()
    entity2id = {}
    id2entity = {}

    with open('../data/{}/entities.dict'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            idx, entity = line.strip().split('\t')
            entity2id[entity] = int(idx)
            id2entity[int(idx)] = entity

    with open('../data/{}/train.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            triples.add('{}\t{}\t{}'.format(h, r, t))

    triples_valid = []
    with open('../data/{}/valid.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            triple = '{}\t{}\t{}'.format(h, r, t)
            triples.add(triple)
            triples_valid.append(triple)
    triples_test = []
    with open('../data/{}/test.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            triple = '{}\t{}\t{}'.format(h, r, t)
            triples.add(triple)
            triples_test.append(triple)
    triple_list_valid = set()
    with open('../data/{}/triple_classification/valid_sampled.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t, label = line.strip().split('\t')
            triple = '{}\t{}\t{}'.format(h, r, t)
            triples.add(triple)
            triple_list_valid.add(triple + '\t' + label)

    triple_list_test = set()
    with open('../data/{}/triple_classification/test_sampled.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t, label = line.strip().split('\t')
            triple = '{}\t{}\t{}'.format(h, r, t)
            triples.add(triple)
            triple_list_test.add(triple + '\t' + label)

    for triple in triple_list_valid:
        h, r, t, label = triple.split('\t')
        if r in valid_data:
            valid_data[r].append((h, t, label))
        else:
            valid_data[r] = [(h, t, label)]

    for triple in triple_list_test:
        h, r, t, label = triple.split('\t')
        if r in test_data:
            test_data[r].append((h, t, label))
        else:
            test_data[r] = [(h, t, label)]

    overall_acc_valid = 0
    thrs = {}
    for relation in valid_data:
        exp_dir = '../exps_{}/{}-{}-ori'.format(dataset, dataset, relation)
        option = Option(exp_dir)
        train_kg, entity2id, relation2id, triple2id = load_kg_form_pkl('{}/'.format(exp_dir), relation.replace('/', '|'))
        results_valid = evaluate(relation, entity2id, triple2id, relation2id, train_kg, valid_data[relation], option,
                 '{}/model_{}.pt'.format(exp_dir, relation.replace('/', '|'), relation.replace('/', '|')))
        best_acc, thr = search_thr(results_valid)
        thrs[relation] = thr
        overall_acc_valid += best_acc / len(valid_data)
        print(relation, thr, best_acc)

    y_true_overall = []
    y_pred_overall = []
    for relation in test_data:
        exp_dir = '../exps_{}/{}-{}-ori'.format(dataset, dataset, relation)
        option = Option(exp_dir)
        train_kg, entity2id, relation2id, triple2id = load_kg_form_pkl('{}/'.format(exp_dir), relation.replace('/', '|'))
        results_test = evaluate(relation, entity2id, triple2id, relation2id, train_kg, test_data[relation], option,
                                 '{}/model_{}.pt'.format(exp_dir, relation.replace('/', '|'),
                                                         relation.replace('/', '|')))
        thr = 0
        if relation in thrs: thr = thrs[relation]
        y_true, y_pred = apply_thr(results_test, thr)
        y_true_overall.extend(y_true)
        y_pred_overall.extend(y_pred)
    overall_acc = metrics.accuracy_score(y_true_overall, y_pred_overall)
    precision = metrics.precision_score(y_true_overall, y_pred_overall)
    recall = metrics.recall_score(y_true_overall, y_pred_overall)
    f1 = metrics.f1_score(y_true_overall, y_pred_overall)
    print(overall_acc, precision, recall, f1)
    print(metrics.classification_report(y_true_overall, y_pred_overall))
