import argparse
import time
import random
import json
import pickle as pkl
import torch
import os
import numpy as np
from model_type import TCLMModel
from tqdm import tqdm
from dataset_type import Dataset

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        if not os.path.exists(self.exps_dir):
            os.mkdir(self.exps_dir)
        flag = '-ori'
        self.exp_dir = os.path.join(self.exps_dir, self.exp_name + '-' + self.target_relation.replace('/', '|') + flag)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        with open(os.path.join(self.exp_dir, "option.txt"), "w") as f:
            json.dump(self.__dict__, f, indent=1)
        return True

class Option_test(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)

def set_seed(option):
    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)
    os.environ['PYTHONHASHSEED'] = str(option.seed)
    if option.use_gpu: torch.cuda.manual_seed_all(option.seed)

def save_data(target_relation, kg, entity2id, relation2id, entity_type, type2id, triple2id, file_path=None):
    print(len(kg))
    with open(os.path.join(file_path, 'kg_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(kg, fw)
    with open(os.path.join(file_path, 'entity2id_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(entity2id, fw)
    with open(os.path.join(file_path, 'relation2id_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(relation2id, fw)
    with open(os.path.join(file_path, 'entity_type_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(entity_type, fw)
    with open(os.path.join(file_path, 'type2id_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(type2id, fw)
    with open(os.path.join(file_path, 'triple2id_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(triple2id, fw)

def build_graph(kg):
    graph = {}
    graph_entity = {}
    relation_tail = {}
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
        if t not in relation_tail: relation_tail[t] = len(relation_tail)
    return graph, graph_entity, relation_tail

def get_type_matrix(entity_type, entity2id, type2id):
    i_x = []
    i_y = []
    v = []
    for entity in entity_type:
        i_x.append(entity)
        i_y.append(entity_type[entity])
        v.append(1)
    return torch.sparse.FloatTensor(torch.LongTensor([i_x, i_y]), torch.FloatTensor(v),
                                    torch.Size([len(entity2id), len(type2id)]))

def create_graph(kg, entity2id, relation2id, triple2id, use_gpu):
    i_x_h = []
    i_y_h = []
    v_h = []
    i_x_t = []
    i_y_t = []
    v_t = []
    i_x_r = []
    i_y_r = []
    v_r = []

    for triple in kg:
        x, r, y = triple.get_triple()
        triple_index = '{}\t{}\t{}'.format(x, r, y)
        relation = relation2id[r]
        i_x_r.append(triple2id[triple_index])
        i_y_r.append(relation)
        v_r.append(1)
        i_x_h.append(entity2id[x])
        i_y_h.append(triple2id[triple_index])
        v_h.append(1)
        i_x_t.append(triple2id[triple_index])
        i_y_t.append(entity2id[y])
        v_t.append(1)

    i = torch.LongTensor([i_x_h, i_y_h])
    v = torch.FloatTensor(v_h)
    e2triple = torch.sparse.FloatTensor(i, v, torch.Size([len(entity2id), len(triple2id)]))

    i = torch.LongTensor([i_x_t, i_y_t])
    v = torch.FloatTensor(v_t)
    triple2e = torch.sparse.FloatTensor(i, v, torch.Size([len(triple2id), len(entity2id)]))

    i = torch.LongTensor([i_x_r, i_y_r])
    v = torch.FloatTensor(v_r)
    triple2r = torch.sparse.FloatTensor(i, v, torch.Size([len(triple2id), len(relation2id)]))

    if use_gpu:
        e2triple = e2triple.cuda()
        triple2e = triple2e.cuda()
        triple2r = triple2r.cuda()

    return e2triple.coalesce(), triple2e.coalesce(), triple2r.coalesce()

def load_data(data_path, target_relation):
    data = {}
    with open(data_path, mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split()
            if len(items) != 3: continue
            h, r, t = items
            if r != target_relation: continue
            if r in data:
                data[r].append((h, r, t))
            else:
                data[r] = [(h, r, t)]
            if 'INV' + r in data:
                data['INV' + r].append((t, 'INV' + r, h))
            else:
                data['INV' + r] = [(t, 'INV' + r, h)]
    return data

def extend_graph(graph_entity, valid_data, test_data):
    for r in valid_data:
        for h, _, t in valid_data[r]:
            if h not in graph_entity:
                graph_entity[h] = {t: [r]}
            else:
                if t in graph_entity[h]:
                    graph_entity[h][t].append(r)
                else:
                    graph_entity[h][t] = [r]

    for r in test_data:
        for h, _, t in test_data[r]:
            if h not in graph_entity:
                graph_entity[h] = {t: [r]}
            else:
                if t in graph_entity[h]:
                    graph_entity[h][t].append(r)
                else:
                    graph_entity[h][t] = [r]

def get_indices(matrix_all):
    indices_all = {}
    indices = matrix_all.indices()
    for i in range(indices.shape[-1]):
        index = indices[:, i]
        flag = '{}\t{}'.format(index[0], index[1])
        indices_all[flag] = i
    return indices_all

def mask_data(matrix, triples, indices, entity2id, triple2id, score, dim):
    values = matrix.values()
    for h, r, t in triples:
        triple = '{}\t{}\t{}'.format(h, r, t)
        triple_id = triple2id[triple]
        if dim == 0:
            flag = '{}\t{}'.format(triple_id, entity2id[t])
        else:
            flag = '{}\t{}'.format(entity2id[h], triple_id)
        index = indices[flag]
        values[index] = score

def valid_process(valid_data, dataset, model, e2triple, triple2e, triple2r, graph_entity, option, raw=False, name='Valid'):
    model.eval()
    mrr = 0
    hit_1 = 0
    hit_3 = 0
    hit_10 = 0
    count = 0
    with torch.no_grad():
        for target_relation in tqdm(valid_data):
            id2head = {}
            head2id = {}
            for h, r, t in valid_data[target_relation]:
                if h not in head2id:
                    head2id[h] = len(head2id)
                    id2head[head2id[h]] = h
            if len(head2id) == 0: return
            if len(head2id) % option.batch_size == 0:
                batch_num = int(len(head2id) / option.batch_size)
            else:
                batch_num = int(len(head2id) / option.batch_size) + 1
            flag = target_relation.startswith('INV')
            total_states = []
            for i in range(batch_num):
                input_x = []
                cur_szie = option.batch_size
                if i == batch_num - 1:
                    cur_szie = len(head2id) - i * option.batch_size
                entity2id_head = {}
                for j in range(cur_szie):
                    entity2id_head[id2head[i * option.batch_size + j]] = j
                for x in entity2id_head.keys():
                    input_x.append(dataset.entity2id[x])

                input_x = torch.LongTensor(input_x)
                input_x = torch.nn.functional.one_hot(input_x, len(dataset.entity2id)).float().to_sparse()
                if option.use_gpu: input_x = input_x.cuda()
                state = model(input_x, e2triple, triple2e, triple2r, flag, is_training=False)
                total_states.append(state.cpu().detach())
            total_states = torch.cat(total_states, dim=0)
            # print('total_states', total_states)
            for h, r, t in valid_data[target_relation]:
                truth_score_ori = total_states[head2id[h]][dataset.entity2id[t]]
                # index_head = torch.argsort(total_states[:, tail2id[t]], descending=True).cpu().numpy()
                if raw:
                    rank = torch.sum((total_states[head2id[h], :] >= truth_score_ori).int())
                else:
                    scores_head = total_states[head2id[h], :].clone()
                    for tail in graph_entity[h]:
                        if r in graph_entity[h][tail]: scores_head[dataset.entity2id[tail]] = -1e20
                    n = torch.sum((scores_head == truth_score_ori).int()) + 1
                    m = torch.sum((scores_head > truth_score_ori).int())
                    rank = m + (n + 1) / 2

                mrr += 1 / rank
                if rank <= 1:
                    hit_1 += 1

                if rank <= 3:
                    hit_3 += 1

                if rank <= 10:
                    hit_10 += 1
                count += 1
    if count > 0:
        mrr /= count
        hit_1 /= count
        hit_3 /= count
        hit_10 /= count
    print('{} Count:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(name, count, mrr, hit_1, hit_3, hit_10))
    return mrr, hit_1, hit_3, hit_10

def train(dataset, option):
    print('Current Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Modeling target relation: {}'.format(option.target_relation))
    print('Entity Num:', len(dataset.entity2id))
    print('Relation Num:', len(dataset.relation2id))
    print('Train KG Size:', len(dataset.kg))
    graph_entity = dataset.graph_entity

    model = TCLMModel(len(dataset.relation2id), option.step, option.length, len(dataset.entity2id),
                  len(dataset.type2id), option.tau_1, option.tau_2, option.use_gpu, option.dropout)
    e2triple, triple2e, triple2r = create_graph(dataset.kg, dataset.entity2id, dataset.relation2id,
                                                dataset.triple2id, option.use_gpu)
    indices_e2triple = get_indices(e2triple)
    indices_triple2e = get_indices(triple2e)

    type_matrix = get_type_matrix(dataset.entity_type, dataset.entity2id, dataset.type2id)
    if option.use_gpu: type_matrix = type_matrix.cuda()

    if option.use_gpu: model = model.cuda()
    for parameter in model.parameters():
        print(parameter)

    optimizer = torch.optim.Adam(model.parameters(), lr=option.learning_rate)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=option.learning_rate * 0.8)

    end_flag = False
    saved_flag = False
    for e in range(option.max_epoch):
        model.train()
        total_loss = 0
        if end_flag: break
        for k, batch in enumerate(dataset.batch_iter()):
            triples, entity_head, flag = batch
            mask_data(e2triple, triples, indices_e2triple, dataset.entity2id, dataset.triple2id, 0, 1)
            mask_data(triple2e, triples, indices_triple2e, dataset.entity2id, dataset.triple2id, 0, 0)
            loss = 0
            input_x = []
            for x in entity_head.keys():
                input_x.append(dataset.entity2id[x])

            input_x = torch.LongTensor(input_x)
            input_x = torch.nn.functional.one_hot(input_x, len(dataset.entity2id)).float().to_sparse()
            if option.use_gpu: input_x = input_x.cuda()
            state = model(input_x, type_matrix, e2triple, triple2e, triple2r, flag, is_training=True)

            for x, r, y in triples:
                logit_mask = torch.zeros([len(dataset.entity2id)])
                for ent in dataset.graph[x][r]:
                    if ent == y: continue
                    logit_mask[dataset.entity2id[ent]] = 1
                loss += model.log_loss(torch.unsqueeze(state[entity_head[x]], dim=0),
                                       dataset.entity2id[y], logit_mask)
            print('Epoch: {}, Batch: {}, Loss: {}'.format(e, k, loss.item()))
            total_loss += loss.item()
            if loss > option.threshold:
                loss.backward()
                optimizer.step()
                sch.step(loss)
                optimizer.zero_grad()
            mask_data(e2triple, triples, indices_e2triple, dataset.entity2id, dataset.triple2id, 1, 1)
            mask_data(triple2e, triples, indices_triple2e, dataset.entity2id, dataset.triple2id, 1, 0)
        print('Epoch: {}, Total Loss: {}'.format(e, total_loss))
    if not saved_flag: torch.save(model.state_dict(),
                           os.path.join(option.exp_dir, 'model_{}.pt'.format(option.target_relation.replace('/', '|'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--use_gpu',  default=False, action="store_true")
    parser.add_argument('--gpu_id', default=4, type=int)
    # model architecture
    parser.add_argument('--length', default=50, type=int)
    parser.add_argument('--step', default=3, type=int)
    parser.add_argument('--tau_1', default=2, type=float)
    parser.add_argument('--tau_2', default=0.2, type=float)
    parser.add_argument('--lammda', default=0.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--with_constrain', default=False, action="store_true")
    # optimization
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--target_relation', default=None, type=str)
    parser.add_argument('--iteration_per_batch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--early_stop', default=True, action="store_true")
    parser.add_argument('--threshold', default=1e-6, type=float)
    parser.add_argument('--negative_sampling', default=False, action="store_true")
    parser.add_argument('--seed', default=1234, type=int)

    d = vars(parser.parse_args())

    option = Option(d)
    if option.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(option.gpu_id)
    option.tag = time.strftime("%y-%m-%d %H:%M:%S")
    bl = option.save()
    print("Option saved.")
    set_seed(option)
    dataset = Dataset(option)
    save_data(option.target_relation, dataset.kg, dataset.entity2id, dataset.relation2id,
              dataset.entity_type, dataset.type2id, dataset.triple2id, option.exp_dir)
    train(dataset, option)

