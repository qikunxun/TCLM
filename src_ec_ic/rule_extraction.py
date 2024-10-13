import os
import json
import sys
import torch
import pickle as pkl
from model_type import TCLMModel

beam_size = int(sys.argv[4])
thr = 0.6

class Option(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)


def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x


def load_kg_form_pkl(file_path, target_relation):
    with open(file_path + 'kg_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        kg = pkl.load(fd)
    with open(file_path + 'entity2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        entity2id = pkl.load(fd)
    with open(file_path + 'relation2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        relation2id = pkl.load(fd)
    with open(file_path + 'entity_type_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        entity_type = pkl.load(fd)
    with open(file_path + 'type2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        type2id = pkl.load(fd)
    return kg, entity2id, relation2id, entity_type, type2id


def get_beam(indices, beam_size):
    beams = indices[:, :beam_size]
    for l in range(indices.shape[0]):
        index = indices[l]
        j = 0
        for i in range(index.shape[-1]):
            beams[l][j] = index[i]
            j += 1
            if j == beam_size: break

    return beams


def get_states(indices, scores):
    states = torch.zeros(indices.shape)
    for l in range(indices.shape[0]):
        states[l] = torch.index_select(scores[l], -1, indices[l])
    return states


def transform_score(x, T):
    one = torch.autograd.Variable(torch.Tensor([1]))
    zero = torch.autograd.Variable(torch.Tensor([0]).detach())
    return torch.minimum(torch.maximum(x / T, zero), one)


def analysis(id2relation, relation2id, type2id, id2type, option, model_save_path):
    T = option.tau_1
    model = TCLMModel(len(relation2id), option.step, option.length, len(entity2id),
                      len(type2id), option.tau_1, option.tau_2, False)

    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.eval()
    n = len(relation2id)
    scores = torch.softmax(model.w[0], dim=-1)
    scores = torch.log(scores)
    indices_order = torch.argsort(scores, dim=-1, descending=True)
    all_indices = []
    indices = get_beam(indices_order, beam_size)
    states = get_states(indices, scores)
    all_indices.append(indices)
    for t in range(1, option.step):
        scores = torch.softmax(model.w[t], dim=-1)
        scores = torch.log(scores)
        scores = states.unsqueeze(dim=-1) + scores.unsqueeze(dim=1)
        scores = scores.view(option.length, -1)
        indices_order = torch.argsort(scores, dim=-1, descending=True)
        topk_indices = get_beam(indices_order, beam_size)
        states = get_states(topk_indices, scores)
        all_indices.append(topk_indices)
    outputs = torch.zeros(option.length, option.step, beam_size).long()
    p = torch.zeros(option.length, beam_size).long()
    for beam in range(beam_size):
        p[:, beam] = beam
    for t in range(option.step):
        for l in range(option.length):
            for beam in range(beam_size):
                if p[l][beam] >= all_indices[t][l].shape[0]: continue
                c = int(all_indices[t][l][p[l][beam]] % (n + 1))
                outputs[l][t][beam] = c
                p_new = int(all_indices[t][l][p[l][beam]] / (n + 1))
                p[l][beam] = p_new

    hidden_rules = []
    for l in range(option.length):
        rule_and = '('
        for t in range(-1, option.step):
            if t > -1:
                h_ = model.h[t][l]
                h_type = model.h_type[t][l]
                alpha = model.alpha[t][l]
                beta = model.beta[t][l]
            else:
                h_ = model.h_x[l]
                alpha = model.alpha_x[l]
                beta = model.beta_x[l]
                h_type = model.h_x_type[l]
            rule = '('
            rule_type = '('
            if transform_score(beta, T) >= thr:
                left = 'z_{}'.format(t)
                if t == -1: left = 'x'
                if t == option.step - 1: left = 'y'
                count = 0
                for r in range(len(relation2id) - 1):
                    if transform_score(h_[r], T) >= thr:
                        if count > 0: rule += ' ∨ '
                        tmp = id2relation[r].split('/')[-1]
                        rule += '{}({}, {})'.format(tmp, left, 'u_{}'.format(count))
                        count += 1
                rule += ')'
            if transform_score(alpha, T) >= thr:
                left = 'z_{}'.format(t)
                if t == -1: left = 'x'
                if t == option.step - 1: left = 'y'
                count = 0
                for c in range(len(type2id)):
                    if transform_score(h_type[c], T) >= thr:
                        if count > 0: rule_type += ' ∨ '
                        tmp = id2type[c].split('/')[-1]
                        rule_type += '{}({})'.format(tmp, left)
                        count += 1
                rule_type += ')'
            rule_tmp = ''
            if len(rule) > 2 and len(rule_type) > 2:
                rule_tmp = '({}  ∨  {})'.format(rule, rule_type)
            elif len(rule) > 2:
                rule_tmp = rule
            elif len(rule_type) > 2:
                rule_tmp = rule_type

            if len(rule_tmp) > 2:
                if rule_and.endswith(')'): rule_and += ' ∧ '
                rule_and += rule_tmp
        rule_and += ')'
        hidden_rules.append(rule_and)

    all_rules = []
    for l in range(option.length):
        rule = '{}(x, y)<-'.format(option.target_relation.split('/')[-1])
        rules = [rule] * beam_size
        counts = torch.zeros(option.length, beam_size)
        for beam in range(beam_size):
            y = ''
            for t in range(option.step):
                c = int(outputs[l][t][beam])
                if c < n:
                    tmp = id2relation[c].split('/')[-1]
                    # if c >= n // 2: tmp = 'INV_' + tmp
                    x = 'x'
                    if counts[l][beam] > 0: x = 'z_{}'.format(int(counts[l][beam]) - 1)
                    y = 'z_{}'.format(int(counts[l][beam]))
                    if t == option.step - 1: y = 'y'
                    output = tmp + '({}, {})'.format(x, y)
                    counts[l][beam] += 1
                else:
                    identity = 'x'
                    if t != 0 and y != '': identity = y
                    output = 'Identity({}, {})'.format(identity, identity)
                end = ''
                if t < option.step - 1: end = ' ∧ '
                rules[beam] += output + end
                if t == option.step - 1 and len(hidden_rules[l]) > 2: rules[beam] += ' ∧ ' + hidden_rules[l]
        all_rules.append(rules)

    ids_sort = torch.argsort(model.weight.squeeze(dim=-1), descending=True)
    fw = open('./{}/rules-{}-ori.txt'.format(option.exps_dir, option.target_relation.replace('/', '|')), mode='w')
    for i, ids in enumerate(ids_sort):
        data = {'rank': (i + 1), 'id': int(ids), 'rules': all_rules[int(ids)],
                'weight': float(torch.tanh(model.weight[int(ids)]))}
        fw.write(json.dumps(data, ensure_ascii=False) + '\n')
        print('Rank: {}, id: {}, Weight: {}, Rule: {}'.format((i + 1), ids, float(torch.tanh(model.weight[int(ids)])),
                                                              all_rules[int(ids)]))
    fw.close()


if __name__ == '__main__':
    exps_dir = sys.argv[1]
    exp_name = sys.argv[2]
    target_relation = sys.argv[3].replace('/', '|')
    option = Option('{}/{}-{}-{}/'.format(exps_dir, exp_name, target_relation, 'ori'))
    train_kg, entity2id, relation2id, entity_type, type2id = load_kg_form_pkl('{}/'.format(option.exp_dir),
                                                                              option.target_relation.replace('/', '|'))
    id2relation = reverse(relation2id)
    id2type = reverse(type2id)
    analysis(id2relation, relation2id, type2id, id2type, option,
             '{}/model_{}.pt'.format(option.exp_dir, option.target_relation.replace('/', '|')))
