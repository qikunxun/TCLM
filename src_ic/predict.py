import os
import json
import sys
import torch
import pickle as pkl
from model import TCLMModel
from tqdm import tqdm
from main import create_graph

batch_size = int(sys.argv[2])
inverse = int(sys.argv[3])
if inverse == 1:
    inverse = True
else:
    inverse = False
print(inverse)
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]


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

def save_head(head2id, flag, file_path=None):
    if file_path is None: return
    with open(os.path.join(file_path, 'head2id_{}.pkl'.format(flag)), mode='wb') as fw:
        pkl.dump(head2id, fw)

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


def evaluate(id2entity, entity2id, id2relation, relation2id, triple2id, train_kg, eval_kg, option, model_save_path):
    print('Entity Num:', len(entity2id))
    print('Relation Num:', len(relation2id))
    print('Train KG Size:', len(train_kg))
    print('Eval KG Size:', len(eval_kg))
    model = TCLMModel(len(relation2id), option.step, option.length,
                           len(entity2id), option.tau_1, option.tau_2, option.use_gpu)

    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    # for parameter in model.parameters():
    #     print(parameter)
    if option.use_gpu: model = model.cuda()
    model.eval()
    e2triple, triple2e, triple2r = create_graph(train_kg, entity2id, relation2id, triple2id, option.use_gpu)

    id2head = {}
    head2id = {}
    for h, r, t in tqdm(eval_kg):
        if r != option.target_relation: continue
        if inverse:
            if t not in head2id:
                head2id[t] = len(head2id)
                id2head[head2id[t]] = t
        else:
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
    for i in tqdm(range(batch_num)):
        cur_szie = batch_size
        if i == batch_num - 1:
            cur_szie = len(head2id) - i * batch_size

        entity2id_head = {}
        for j in range(cur_szie):
            entity2id_head[id2head[i * batch_size + j]] = j
        input_x = []
        for x in entity2id_head.keys():
            input_x.append(entity2id[x])
        input_x = torch.LongTensor(input_x)
        input_x = torch.nn.functional.one_hot(input_x, len(entity2id)).float().to_sparse()
        if option.use_gpu: input_x = input_x.cuda()
        state = model(input_x, e2triple, triple2e, triple2r, inverse, is_training=False)

        total_states.append(state.cpu().detach())
    total_states = torch.cat(total_states, dim=0)
    flag = 'ori'
    if inverse: flag = 'inv'
    torch.save(total_states, '{}/state-{}.pt'.format(option.exp_dir, flag))
    save_head(head2id, flag, file_path=option.exp_dir)

if __name__ == '__main__':
    option = Option(sys.argv[1])
    train_kg, entity2id, relation2id, triple2id = load_kg_form_pkl('{}/'.format(option.exp_dir),
                                                                   option.target_relation.replace('/', '|'))
    id2entity = reverse(entity2id)
    id2relation = reverse(relation2id)
    eval_kg = load_kg('{}/test.txt'.format(option.data_dir))
    evaluate(id2entity, entity2id, id2relation, relation2id, triple2id, train_kg, eval_kg, option,
             '{}/model_{}.pt'.format(option.exp_dir, option.target_relation.replace('/', '|'),
                                     option.target_relation.replace('/', '|')))