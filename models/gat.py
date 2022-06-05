import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Simple GAT layer"""

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 leakyrelu_alpha,
                 concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.leakyrelu_alpha = leakyrelu_alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(input_size, output_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.leakyrelu_alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        # h.shape [N, input_size], Wh.shape [N, output_size], N = number of nodes
        e = self._compute_attention_coefficients(Wh)

        # masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        e = torch.where(adj > 0, e, zero_vec)
        # normalized attention coefficients alpha
        alpha = F.softmax(e, dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h_prime = torch.matmul(alpha, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _compute_attention_coefficients(self, Wh):
        # Wh.shape [N, out_feature], self.a.shape [2 x out_feature, 1]
        # Wh1.shape and Wh2.shape [N, 1], e.shape [N, N]
        Wh1 = torch.matmul(Wh, self.a[:self.output_size, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_size:, :])
        e = Wh1 + Wh2.T  # broadcast add
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.input_size) + ' -> ' + str(self.output_size) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):

    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """Sparse version GAT layer"""

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 leakyrelu_alpha,
                 concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.leakyrelu_alpha = leakyrelu_alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(input_size, output_size)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * output_size)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.leakyrelu_alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]),
                                     torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.input_size) + ' -> ' + str(self.output_size) + ')'


class GAT(nn.Module):
    """Dense GAT"""

    def __init__(self, nfeat, nhid, nclass, dropout, leakyrelu_alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat,
                                nhid,
                                dropout=dropout,
                                leakyrelu_alpha=leakyrelu_alpha,
                                concat=True) for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads,
                                           nclass,
                                           dropout=dropout,
                                           leakyrelu_alpha=leakyrelu_alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    """Sparse GAT"""

    def __init__(self, nfeat, nhid, nclass, dropout, leakyrelu_alpha, nheads):
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            SpGraphAttentionLayer(nfeat,
                                  nhid,
                                  dropout=dropout,
                                  leakyrelu_alpha=leakyrelu_alpha,
                                  concat=True) for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             leakyrelu_alpha=leakyrelu_alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = GAT(nfeat=1024,
                nhid=64,
                nclass=10,
                dropout=0.5,
                leakyrelu_alpha=0.2,
                nheads=3)
    model = model.cuda()
    print(model)

    # inputs size [nnode, nfeat], adj size [nnode, nnode]
    inputs = torch.randn(6, 1024).cuda()
    adj = torch.randn(6, 6).cuda()
    y = model(inputs, adj)
    print(y)
