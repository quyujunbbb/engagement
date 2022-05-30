import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import normalize



# ------------------------------------------------------------------------------
# subtraction
all = np.array([0.0142, 0.0773])
b = np.array([0.0451, 0.1750])
a = np.array([0.1235, 0.3690])
v = np.array([0.1567, 0.4184])
wa = np.array([0.0567, 0.2130])
wg = np.array([0.0313, 0.1091])

print(b-all)
print(a-all)
print(v-all)
print(wa-all)
print(wg-all)


# ------------------------------------------------------------------------------
# create random loss
# loss_mse = nn.MSELoss()
# loss_mae = nn.L1Loss()

# y = normalize(torch.randn(5000))
# y_hat = normalize(torch.randn(4, 5000))

# for pred in y_hat:
#     res_mse = loss_mse(pred, y)
#     res_mae = loss_mae(pred, y)
#     print(res_mse, res_mae)

# tensor(2.1133) tensor(1.1514)
# tensor(2.0352) tensor(1.1366)
# tensor(1.9788) tensor(1.1195)
# tensor(1.9618) tensor(1.1111)


# ------------------------------------------------------------------------------
# in classification, the best results
labels = pd.read_csv('data/annotations/labels_class.csv')
y = torch.tensor(labels['overall'].values)
y_hat = torch.tensor(labels['class_value'].values)
print(y)
print(y_hat)

loss_func_mse = nn.MSELoss()
loss_func_mae = nn.L1Loss()

loss_mse = loss_func_mse(y_hat, y)
loss_mae = loss_func_mae(y_hat, y)

print(f'mse={loss_mse:.6f}, mae={loss_mae:.6f}')
# mse=0.000064, mae=0.002340
