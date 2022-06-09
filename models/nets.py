import numpy as np
import torch
from torch import nn, unsqueeze
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


class NonLocal(nn.Module):

    def __init__(self):
        super(NonLocal, self).__init__()

        self.nl = NonLocalBlock(in_channels=4096)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)
        # print(f'batch size: {batch_size}, x shape: {x.shape}')

        out = x.reshape(batch_size, 4, -1, 1024, 7, 7)
        # print(out.shape)  # [batch_size, 4, 4, 1024, 7, 7]
        out = out.reshape(batch_size, 4, -1, 7, 7).permute(0, 2, 1, 3, 4)
        # print(out.shape)  # [batch_size, 4096, 4, 7, 7]
        out = self.nl(out)
        out = self.avgpool(out)
        # print(out.shape)  # [batch_size, 4096, 1, 1, 1]
        out = out.reshape(batch_size, -1).reshape(batch_size, 4, 1024)
        # print(out.shape)  # [batch_size, 4, 1024]

        return out


# ------------------------------------------------------------------------------
# Graph Attention Neural Networks
class GAL(nn.Module):

    def __init__(self, input_size, output_size, dropout, alpha, concat=True):
        super(GAL, self).__init__()
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
            # print('concat')
            return F.elu(h_prime)
        else:
            # print('not concat')
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


class AdaptedGAL(nn.Module):

    def __init__(self, input_size, output_size, dropout, alpha, concat=True):
        super(AdaptedGAL, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(input_size, output_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a0 = nn.Parameter(torch.empty(size=(2 * output_size, 1)))
        nn.init.xavier_uniform_(self.a0.data, gain=1.414)
        self.ai = nn.Parameter(torch.empty(size=(2 * output_size, 1)))
        nn.init.xavier_uniform_(self.ai.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [4, 1024], W: [1024, 64] --> Wh: [4, 64] --> e: [4, 4]
        # alpha: [4, 4] --> h_prime: [4, 64]
        Wh = torch.mm(h, self.W)
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
        # Wh: [4, 64], a: [64*2, 1] --> Wh1: [4, 1], Wh2: [4, 1] --> e: [4, 4]
        Wh0_1 = torch.matmul(Wh, self.a0[:self.output_size, :])
        Wh0_2 = torch.matmul(Wh, self.a0[self.output_size:, :])
        e0 = Wh0_1 + Wh0_2.T

        Whi_1 = torch.matmul(Wh, self.ai[:self.output_size, :])
        Whi_2 = torch.matmul(Wh, self.ai[self.output_size:, :])
        ei = Whi_1 + Whi_2.T

        e = torch.cat((e0[0, :].unsqueeze(0), ei[1:, :]), dim=0)

        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.input_size) + ' --> ' + str(self.output_size) + ')'


class GAT(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 head_num,
                 layer,
                 activation=True,
                 dropout=0.5,
                 alpha=0.2):
        super(GAT, self).__init__()

        if layer == 'GAL':
            self.layer = GAL(input_size,
                             output_size,
                             dropout,
                             alpha,
                             concat=activation)
            self.layer_out = GAL(output_size * head_num,
                                 output_size,
                                 dropout,
                                 alpha,
                                 concat=False)
        elif layer == 'AdaptedGAL':
            self.layer = AdaptedGAL(input_size,
                                    output_size,
                                    dropout,
                                    alpha,
                                    concat=activation)
            self.layer_out = AdaptedGAL(output_size * head_num,
                                        output_size,
                                        dropout,
                                        alpha,
                                        concat=False)

        self.dropout = dropout
        self.fc = nn.Linear(in_features=64, out_features=1)
        self.attentions = [self.layer for _ in range(head_num)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'GraphAttention{i}', attention)

    def forward(self, x):
        batch_size = x.size(0)
        adj = torch.randn(batch_size, 4, 4).to(device='cuda')
        out = torch.ones(batch_size, 4, 64).to(device='cuda')
        # print(out.shape)
        for i in range(batch_size):
            # print(x[i].shape, adj[i].shape)
            temp = F.dropout(x[i], self.dropout, training=self.training)
            # print(temp.shape)
            temp = torch.cat([att(temp, adj[i]) for att in self.attentions],
                             dim=1)
            # print(temp.shape)
            temp = F.dropout(temp, self.dropout, training=self.training)
            # print(temp.shape)
            # temp = F.elu(self.out_att(temp, adj[i]))
            out[i] = self.layer_out(temp, adj[i])
        # print(out.shape)
        out = out[:, 0, :]
        # print(out.shape)
        out = self.fc(out)

        return out


class NonLocalGAT(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 head_num,
                 layer,
                 activation=True,
                 dropout=0.5,
                 alpha=0.2):
        super(NonLocalGAT, self).__init__()

        self.nl = NonLocalBlock(in_channels=4096)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if layer == 'GAL':
            self.layer = GAL(input_size,
                             output_size,
                             dropout,
                             alpha,
                             concat=activation)
            self.layer_out = GAL(output_size * head_num,
                                 output_size,
                                 dropout,
                                 alpha,
                                 concat=False)
        elif layer == 'AdaptedGAL':
            self.layer = AdaptedGAL(input_size,
                                    output_size,
                                    dropout,
                                    alpha,
                                    concat=activation)
            self.layer_out = AdaptedGAL(output_size * head_num,
                                        output_size,
                                        dropout,
                                        alpha,
                                        concat=False)

        self.dropout = dropout
        self.fc = nn.Linear(in_features=64, out_features=1)
        self.attentions = [self.layer for _ in range(head_num)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'GraphAttention{i}', attention)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.reshape(batch_size, 4, -1, 1024, 7, 7)
        x = x.reshape(batch_size, 4, -1, 7, 7).permute(0, 2, 1, 3, 4)
        x = self.nl(x)
        x = self.avgpool(x)
        x = x.reshape(batch_size, -1).reshape(batch_size, 4, 1024)

        adj = torch.randn(batch_size, 4, 4).to(device='cuda')
        out = torch.zeros(batch_size, 4, 64).to(device='cuda')
        for i in range(batch_size):
            temp = F.dropout(x[i], self.dropout, training=self.training)
            temp = torch.cat([att(temp, adj[i]) for att in self.attentions],
                             dim=1)
            temp = F.dropout(temp, self.dropout, training=self.training)
            out[i] = self.layer_out(temp, adj[i])
        out = out[:, 0, :]
        out = self.fc(out)

        return out


if __name__ == '__main__':
    input = np.load('features/roi_features_new/20201222_01/0.npy')
    input = torch.from_numpy(input)
    input = input.unsqueeze(0).to(device='cuda')
    print(f'input shape: {input.shape}')

    # non-local block
    net1 = NonLocal().to(device='cuda')
    out = net1(input)
    print(f'non-local block output shape: {out.shape}')

    # gat
    layer = 'AdaptedGAL'  # GAL or AdaptedGAL
    net2 = GAT(input_size=1024, output_size=64, head_num=3,
               layer=layer).to(device='cuda')
    out = net2(out)
    print(f'gat output shape: {out.shape}')

    # non-local + gat
    net3 = NonLocalGAT(input_size=1024,
                       output_size=64,
                       head_num=3,
                       layer=layer).to(device='cuda')
    out = net3(input)
    print(f'non-local + gat output shape: {out.shape}')

    # params = net3.state_dict()
    # print(params.keys())
    gat_params = [
        'layer.W', 'layer.a0', 'layer.ai', 'layer_out.W', 'layer_out.a0',
        'layer_out.ai', 'GraphAttention0.W', 'GraphAttention0.a0',
        'GraphAttention0.ai', 'GraphAttention1.W', 'GraphAttention1.a0',
        'GraphAttention1.ai', 'GraphAttention2.W', 'GraphAttention2.a0',
        'GraphAttention2.ai'
    ]
    for name, param in net3.named_parameters():
        if param.requires_grad and name in gat_params:
            param.requires_grad = False

    for name, param in net3.named_parameters():
        if param.requires_grad:
            print(name)
