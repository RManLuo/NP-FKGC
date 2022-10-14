from collections import OrderedDict

import torch
import torch.nn.functional as F
from hyper_embedding import *
from torch.autograd import Variable
from torch.distributions import kl_divergence

from embedding import *
from flow import Flow
from relational_path_gnn import RelationalPathGNN


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)


class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True, dropout=self.dropout)
        #self.gru = nn.GRU(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden*2*self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2).cuda()
        #batchnorm = nn.BatchNorm1d(5, affine=False).cuda()
        #attn_weight = batchnorm(attn_weight)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weight)
        context = context.view(-1, self.n_hidden*2*self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        cell_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))  # LSTM
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_cell_state)      # change log

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)    


class EmbeddingLearner(nn.Module):
    def __init__(self, emb_dim, z_dim, out_size):
        super(EmbeddingLearner, self).__init__()
        self.head_encoder= nn.Linear(emb_dim, emb_dim)
        self.tail_encoder= nn.Linear(emb_dim, emb_dim)
        self.dr = nn.Linear(z_dim, 1)

    def forward(self, h, t, r, pos_num, z):					# revise
        # z = torch.nn.functional.normalize(z, dim=-1)
        d_r = self.dr(z)
        z = z.unsqueeze(2)
        h = h + self.head_encoder(z)
        t = t + self.tail_encoder(z)
        tmp_score = torch.norm(h + r - t, 2, -1)
        score = - torch.norm(tmp_score - d_r ** 2, 2, -1)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


def save_grad(grad):
    global grad_norm
    grad_norm = grad


class MetaR(nn.Module):
    def __init__(self, g, dataset, parameter, num_symbols, embed = None):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.num_rel = len(self.rel2id)
        self.embedding = Embedding(dataset, parameter)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0.5)
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx = num_symbols)
        self.num_hidden1 = 500
        self.num_hidden2 = 200
        self.lstm_dim = parameter['lstm_hiddendim']
        self.lstm_layer = parameter['lstm_layers']
        self.np_flow = parameter['flow']
        
        self.r_path_gnn = RelationalPathGNN(g, dataset['ent2id'], len(dataset['rel2emb']), parameter)

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
        self.symbol_emb.weight.requires_grad = False

        if parameter['dataset'] == 'Wiki-One':
            self.r_dim = self.z_dim = 50
            self.relation_learner = LSTM_attn(embed_size=50, n_hidden=100, out_size=50,layers=2, dropout=0.5)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, r_dim=self.z_dim, dropout_p=self.dropout_p)
            self.embedding_learner = EmbeddingLearner(50, self.z_dim, 50)

        elif parameter['dataset'] == 'NELL-One':
            self.r_dim = self.z_dim = 100
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=self.lstm_dim, out_size=100, layers=self.lstm_layer, dropout=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, r_dim=self.z_dim, dropout_p=self.dropout_p)
            self.embedding_learner = EmbeddingLearner(100, self.z_dim, 100)
        elif parameter['dataset'] == 'FB15K-One':
            self.r_dim = self.z_dim = 100
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=self.lstm_dim, out_size=100, layers=self.lstm_layer, dropout=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, r_dim=self.z_dim, dropout_p=self.dropout_p)
            self.embedding_learner = EmbeddingLearner(100, self.z_dim, 100)
        if self.np_flow != 'none':
            self.flows = Flow(self.z_dim, parameter['flow'], parameter['K'])
            
        self.xy_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim)
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        
    def eval_reset(self):
        self.eval_query = None
        self.eval_z = None
        self.eval_rel = None
        self.is_reset = True

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def eval_support(self, support, support_negative, query):
        support, support_negative, query = self.r_path_gnn(support), self.r_path_gnn(support_negative), self.r_path_gnn(query)
        num_q = query.shape[1]
        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)
        support_pos_r = self.latent_encoder(support, 1)
        support_neg_r = self.latent_encoder(support_negative, 0)
        target_r = torch.cat([support_pos_r, support_neg_r], dim=1)
        context_dist = None
        target_dist = self.xy_to_mu_sigma(target_r)
        z = target_dist.sample()
        if self.np_flow != 'none':
            z, _ = self.flows(z, target_dist)
        rel = self.relation_learner(support_few)
        return query, z, rel
    
    def eval_forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        support, support_negative, query, negative = task
        negative = self.r_path_gnn(negative)
        if self.is_reset:
            query, z, rel = self.eval_support(support, support_negative, query)
            self.eval_query = query
            self.eval_z = z
            self.eval_rel = rel
            self.is_reset = False
        else:
            query = self.eval_query
            z = self.eval_z
            rel = self.eval_rel
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)
        return p_score, n_score
            
    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        # transfer task string into embedding
        # support, support_negative, query, negative = [self.embedding(t) for t in task]
        support, support_negative, query, negative = [self.r_path_gnn(t) for t in task]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)
        rel = self.relation_learner(support_few)
        
        # Encoder
        if iseval or istest:
            support_pos_r = self.latent_encoder(support, 1)
            support_neg_r = self.latent_encoder(support_negative, 0)
            target_r = torch.cat([support_pos_r, support_neg_r], dim=1)
            context_dist = None
            target_dist = self.xy_to_mu_sigma(target_r)
            z = target_dist.sample()
            if self.np_flow != 'none':
                z, _ = self.flows(z, target_dist)
        else:
            query_pos_r = self.latent_encoder(query, 1)
            query_neg_r = self.latent_encoder(negative, 0)
            support_pos_r = self.latent_encoder(support, 1)
            support_neg_r = self.latent_encoder(support_negative, 0)
            context_r = torch.cat([support_pos_r, support_neg_r], dim=1)
            target_r = torch.cat([support_pos_r, support_neg_r, query_pos_r, query_neg_r], dim=1)
            context_dist = self.xy_to_mu_sigma(context_r)
            target_dist = self.xy_to_mu_sigma(target_r)
            z = target_dist.rsample()
            if self.np_flow != 'none':
                z, kld = self.flows(z, target_dist, context_dist)
            else:
                kld =  kl_divergence(target_dist, context_dist).sum(-1)
        
        rel = self.relation_learner(support_few)
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)

        if iseval:
            return p_score, n_score
        else:
            return p_score, n_score, kld
    
class LatentEncoder(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, r_dim=100, dropout_p=0.5):
        super(LatentEncoder, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size+1, num_hidden1)),
            #('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            #('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, r_dim)),
            #('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs, y):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1) # (B, few, dim * 2)
        if y == 1:
            label = torch.ones(size[0], size[1], 1).to(inputs)
        else:
            label = torch.zeros(size[0], size[1], 1).to(inputs)
        x = torch.cat([x, label], dim=-1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)

        return x # (B, few, r_dim)

class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def aggregate(self, r):
        return torch.mean(r, dim=1)
    
    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, few, r_dim)
        """
        r = self.aggregate(r)
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)
