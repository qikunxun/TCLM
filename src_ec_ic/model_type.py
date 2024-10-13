import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda')
class TCLMModel(nn.Module):
    def __init__(self, n, T, L, N, K, tau_1=10, tau_2=0.2, use_gpu=False, dropout=0.1):
        super(TCLMModel, self).__init__()
        self.T = T
        self.L = L
        self.N = N
        self.n = n
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.K = K
        self.w = nn.Parameter(torch.Tensor(self.T, self.L, self.n))
        nn.init.kaiming_uniform_(self.w, a=np.sqrt(5))
        self.weight = nn.Parameter(torch.Tensor(self.L, 1))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))


        self.h = torch.ones(self.T, self.L, self.n - 1)
        self.h_x = torch.ones(self.L, self.n - 1)
        self.h = nn.Parameter(self.h)
        self.h_x = nn.Parameter(self.h_x)

        self.h_type = torch.ones(self.T, self.L, self.K)
        self.h_type = nn.Parameter(self.h_type)
        nn.init.kaiming_uniform_(self.h_type, a=np.sqrt(5))
        self.h_x_type = torch.ones(self.L, self.K)
        self.h_x_type = nn.Parameter(self.h_x_type)
        nn.init.kaiming_uniform_(self.h_x_type, a=np.sqrt(5))

        self.alpha = torch.ones(self.T, self.L)
        self.alpha = nn.Parameter(self.alpha)
        self.beta = torch.ones(self.T, self.L)
        self.beta = nn.Parameter(self.beta)

        self.alpha_x = torch.ones(self.L)
        self.alpha_x = nn.Parameter(self.alpha_x)
        self.beta_x = torch.ones(self.L)
        self.beta_x = nn.Parameter(self.beta_x)
        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, type, e2triple, triple2e, triple2r, flag, is_training=False):
        batch_size = input_x.shape[0]
        N = triple2e.shape[0]
        E = triple2e.shape[1]
        x_ori = torch.sparse.mm(input_x, e2triple)  # [b, N]
        e2r = torch.sparse.mm(triple2e.transpose(1, 0), triple2r)  # [E, n]
        states = []
        for t in range(self.T):
            w_probs = self.w[t]
            h_probs = self.h[t]
            h_x_probs = self.h_x
            h_type_probs = self.h_type[t]
            h_x_type_probs = self.h_x_type

            alpha = self.alpha[t]
            beta = self.beta[t]
            alpha_x = self.alpha_x
            beta_x = self.beta_x
            if flag:
                w_probs = self.w[self.T - 1 - t]
                h_x_probs = self.h[-1]
                h_x_type_probs = self.h_type[-1]
                alpha_x = self.alpha[-1]
                beta_x = self.beta[-1]
                if t == self.T - 1:
                    h_probs = self.h_x
                    h_type_probs = self.h_x_type
                    alpha = self.alpha_x
                    beta = self.beta_x
                else:
                    h_probs = self.h[self.T - 2 - t]
                    h_type_probs = self.h_type[self.T - 2 - t]
                    alpha = self.alpha[self.T - 2 - t]
                    beta = self.beta[self.T - 2 - t]
            w_probs = torch.softmax(w_probs, dim=-1)
            if flag: w_probs = torch.cat([w_probs[:, (self.n - 1) // 2: -1],
                                          w_probs[:, :(self.n - 1) // 2], w_probs[:, -1:]], dim=-1)
            hidden = e2r
            hidden = self.activation(hidden.to_dense())  # [E, n]
            hidden = hidden[:, :-1].to_sparse()     # [E, n]
            h_probs = self.activation(h_probs / self.tau_1)
            h_type_probs = self.activation(h_type_probs / self.tau_1)
            alpha = self.activation(alpha / self.tau_1)
            beta = self.activation(beta / self.tau_1)

            hidden = self.activation(alpha.unsqueeze(dim=0) * torch.sparse.mm(type, torch.permute(h_type_probs, (1, 0)))
                                     + beta.unsqueeze(dim=0) * torch.sparse.mm(hidden, torch.permute(h_probs, (1, 0))))
            gate = 1 - self.activation(alpha + beta)
            hidden = hidden + torch.ones_like(hidden) * gate.unsqueeze(dim=0)
            hidden = hidden.transpose(1, 0)


            hidden_x = torch.sparse.mm(x_ori, triple2r)   # [b, n]
            hidden_x = self.activation(hidden_x.to_dense())   # [b, n]
            hidden_x = torch.cat([hidden_x[:, (self.n - 1) // 2: -1], hidden_x[:, :(self.n - 1) // 2]], dim=-1)
            h_x_probs = self.activation(h_x_probs / self.tau_1)
            h_x_type_probs = self.activation(h_x_type_probs / self.tau_1)
            alpha_x = self.activation(alpha_x / self.tau_1)
            beta_x = self.activation(beta_x / self.tau_1)

            hidden_type_x = torch.sparse.mm(type, torch.permute(h_x_type_probs, (1, 0))) # [E, L]
            diag = input_x  # [b, E]
            hidden_type_x = torch.sparse.mm(diag, hidden_type_x) # [b, L]

            hidden_x = self.activation(alpha.unsqueeze(dim=0) * hidden_type_x +
                                       beta.unsqueeze(dim=0) * torch.mm(hidden_x, torch.permute(h_x_probs, (1, 0))))
            gate_x = 1 - self.activation(alpha_x + beta_x)
            hidden_x = hidden_x + torch.ones_like(hidden_x) * gate_x.unsqueeze(dim=0)

            if t == 0:
                x = x_ori.to_dense()  # [b, N]
                w = w_probs  # [L, r]
                s = torch.sparse.mm(triple2r, w.transpose(1, 0))
                s = torch.einsum('bm,ml->blm', x, s)  # [b, L, N]
                s = torch.sparse.mm(s.reshape(-1, N), triple2e).view(batch_size, self.L, E)  # [b, L, E]
                s = s * hidden.unsqueeze(dim=0) * hidden_x.unsqueeze(dim=2)  # [b, L, E]
                if is_training: s = self.dropout(s)
            if t >= 1:
                x = states[-1]  # [b, L, E]
                x = torch.sparse.mm(x.reshape(-1, E), e2triple).view(batch_size, self.L, N)  # [b, L, N]
                w = w_probs  # [L, r]
                s = torch.sparse.mm(triple2r, w.transpose(1, 0))
                s = torch.einsum('blm,ml->blm', x, s)  # [b, L, N]
                s = torch.sparse.mm(s.reshape(-1, N), triple2e).view(batch_size, self.L, E)  # [b, L, E]
                s = s * hidden.unsqueeze(dim=0)  # [b, L, E]
                if is_training: s = self.dropout(s)
            states.append(s)
        state = states[-1]

        weight = self.weight
        weight = torch.tanh(weight)
        s = state
        if is_training: s = self.dropout(s)
        s = torch.einsum('blm,lk->bmk', s, weight).squeeze(dim=-1)  # [b*E, 1]
        return s



    def log_loss(self, p_score, label, logit_mask, thr=1e-20):
        one_hot = F.one_hot(torch.LongTensor([label]), p_score.shape[-1])
        if self.use_gpu:
            one_hot = one_hot.to(device)
            logit_mask = logit_mask.to(device)
        p_score = p_score - 1e30 * logit_mask.unsqueeze(dim=0)
        loss = -torch.sum(
            one_hot * torch.log(torch.maximum(F.softmax(p_score / self.tau_2, dim=-1), torch.ones_like(p_score) * thr)),
            dim=-1)
        return loss

    def activation(self, x):
        one = torch.autograd.Variable(torch.Tensor([1]))
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu:
            one = one.to(device)
            zero = zero.to(device)
        return torch.minimum(torch.maximum(x, zero), one)
