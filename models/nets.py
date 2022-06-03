import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        self.nl = NonLocalBlock(in_channels=1024)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.nl(x)  # (bs, 1024, 8, 7, 7)
        print(x.shape)
        x = x.permute(0, 2, 1, 3, 4)  # (bs, 8, 1024, 7, 7)
        print(x.shape)
        x = self.avgpool(x).view(batch_size, -1)  # (bs, 1024)
        print(x.shape)
        # x = self.fc(x)

        return x


if __name__ == '__main__':
    input = np.load('features/roi_features_new/20201222_01/0.npy')
    input = torch.from_numpy(input)
    input = input.permute(1, 0, 2, 3)
    input = input.unsqueeze(0).to(device='cuda')
    print(f'input roi shape: {input.shape}')

    net = MyModel().to(device='cuda')
    output = net(input)
    print(f'output shape: {output.shape}')
