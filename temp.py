import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter


def watch_session_info():
    # print data
    data_info = torch.load('data/data_info.pt')
    print(data_info)
    # count clip number
    count = 0
    for i in range(len(data_info)):
        count += data_info[i, 0]
    print(count)


def export_tensorboardx():
    timestamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
    writer = SummaryWriter('runs')

    res = []
    for x in range(50):
        y1 = np.sin(x)
        y2 = np.cos(x)
        temp = [x, y1, y2]
        res.append(temp)
        writer.add_scalars(timestamp, {'y1': y1, 'y2': y2}, x)

    writer.close()

    output_path = f'runs/{timestamp}/res.csv'
    res = pd.DataFrame(res, columns=['x', 'y1', 'y2'])
    res.to_csv(output_path, index=False)

    fig, ax = plt.subplots()
    ax.plot(res['x'], res['y1'], res['x'], res['y2'])
    ax.set(xlabel='x', ylabel='y', title='test')
    ax.grid()
    fig.savefig(f'runs/{timestamp}/test.png')
    plt.show()


if __name__ == '__main__':
    pass
