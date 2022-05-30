from torch import nn
from torchinfo import summary

from models.non_local import NONLocalBlock3D


class Nonlocal_FC3_Reg(nn.Module):

    def __init__(self):
        super(Nonlocal_FC3_Reg, self).__init__()

        self.nl = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 7, 7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.nl(x)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


class Nonlocal_FC1_Reg(nn.Module):

    def __init__(self):
        super(Nonlocal_FC1_Reg, self).__init__()

        self.nl = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 7, 7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=1))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.nl(x)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


class Nonlocal_FC1_Class(nn.Module):

    def __init__(self):
        super(Nonlocal_FC1_Class, self).__init__()

        self.nl = NONLocalBlock3D(in_channels=1024)
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 7, 7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=13))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.nl(x)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


class FC3_Reg(nn.Module):

    def __init__(self):
        super(FC3_Reg, self).__init__()

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 7, 7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(in_features=512, out_features=256),
                                nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(in_features=256, out_features=1))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


class FC1_Reg(nn.Module):

    def __init__(self):
        super(FC1_Reg, self).__init__()

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 7, 7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=1))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)

        return x


def get_network_structure(model, input):
    summary(model, input_size=input)


if __name__ == '__main__':
    net = FC1_Reg()
    input = (16, 1024, 4, 7, 7)
    get_network_structure(net, input)
