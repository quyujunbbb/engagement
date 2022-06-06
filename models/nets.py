import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ------------------------------------------------------------------------------
# Non-local Neural Networks
class NonLocalBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(NonLocalBlock, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        x             : (b, c, t, h, w)
        return_nl_map : if True return z, nl_map, else only return z.
        """
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


# ------------------------------------------------------------------------------
# Graph Attention Neural Networks
class GraphAttentionLayer(nn.Module):

    def __init__(self, input_size, output_size, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(input_size, output_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

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
            print('concat')
            return F.elu(h_prime)
        else:
            print('not concat')
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
            self.input_size) + ' --> ' + str(self.output_size) + ')'


class GAT(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, head_num, dropout,
                 alpha):
        super(GAT, self).__init__()

        self.dropout = dropout
        self.attentions = [
            GraphAttentionLayer(input_size,
                                hidden_size,
                                dropout=dropout,
                                alpha=alpha,
                                concat=True) for _ in range(head_num)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_size * head_num,
                                           output_size,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        print(x.shape)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        print(x.shape)
        x = F.elu(self.out_att(x, adj))
        print(x.shape)
        return F.log_softmax(x, dim=1)


class MyGAT(nn.Module):

    def __init__(self, input_size, output_size, head_num, dropout, alpha):
        super(MyGAT, self).__init__()

        self.dropout = dropout
        self.attentions = [
            GraphAttentionLayer(input_size,
                                output_size,
                                dropout=dropout,
                                alpha=alpha,
                                concat=False) for _ in range(head_num)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'GraphAttention{i}', attention)
        self.out_att = GraphAttentionLayer(output_size * head_num,
                                           output_size,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        print(x.shape)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        print(x.shape)
        x = F.elu(self.out_att(x, adj))
        print(x.shape)
        return F.log_softmax(x, dim=1)


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        self.nl = NonLocalBlock(in_channels=1024)
        self.avgpool = nn.AdaptiveAvgPool3d((8, 1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.nl(x)  # (bs, 1024, 8, 7, 7)
        print(x.shape)
        x = self.avgpool(x)  # (bs, 1024) .view(batch_size, -1)
        print(x.shape)
        x = x.permute(0, 2, 1, 3, 4)  # (bs, 8, 1024, 7, 7)
        print(x.shape)
        # x = self.fc(x)

        return x


if __name__ == '__main__':
    input = np.load('features/roi_features_new/20201222_01/0.npy')
    input = torch.from_numpy(input)
    input = input.permute(1, 0, 2, 3)
    input = input.unsqueeze(0).to(device='cuda')
    print(f'input roi shape: {input.shape}')

    # non-local block
    net = MyModel().to(device='cuda')
    output = net(input)
    print(f'output shape: {output.shape}')

    # gat
    # input size [node_num, input_size], adj size [node_num, node_num]
    input = torch.randn(4, 1024).cuda()
    adj = torch.randn(4, 4).cuda()

    model = MyGAT(input_size=1024,
                  output_size=64,
                  head_num=3,
                  dropout=0.5,
                  alpha=0.2).to(device='cuda')
    print(model)

    y = model(input, adj)
    print(f'y shape: {y.shape}')
