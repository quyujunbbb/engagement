import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.cuda import amp
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary

from models import nets
from utils.bheh import create_bheh


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', type=str)
    parser.add_argument('-d', '--data', dest='dataset', type=str)

    return parser


class Config:
    epoch = 10
    batch_size = 16
    learning_rate = 1e-4
    step_size = 10
    gamma = 0.1
    num_workers = 2
    # use_dropout = True
    # ckpt_root_dir = './checkpoints'
    # output_dir = 'AffectNet_res18'
    # pretrained = './pretrain/checkpoints/out_dir_res18/mv_epoch_17.pt'


def draw_loss(loss_train, loss_test_mse, loss_test_mae, step):
    writer = SummaryWriter('results/runs')
    writer.add_scalars(
        timestamp, {
            'loss_train': loss_train,
            'loss_test_mse': loss_test_mse,
            'loss_test_mae': loss_test_mae
        }, step)
    writer.close()


def save_results(res):
    res = pd.DataFrame(res, columns=['setp', 'train', 'test_mse', 'test_mae'])
    res.to_csv(f'results/csv/{timestamp}.csv', index=False)

    fig, ax = plt.subplots()
    ax.plot(res['setp'], res['train'], res['setp'], res['test_mse'])
    ax.grid()
    fig.savefig(f'results/figs/{timestamp}.png')

    smooth_factor = 0.8
    smooth_train = res['train'].ewm(alpha=(1 - smooth_factor)).mean()
    smooth_test = res['test_mse'].ewm(alpha=(1 - smooth_factor)).mean()
    fig, ax = plt.subplots()
    ax.plot(res['setp'], smooth_train, res['setp'], smooth_test)
    ax.grid()
    fig.savefig(f'results/figs/{timestamp}_smooth.png')


def train(model, cfg, timestamp, output_path):
    # ckpt_save_path = f'{output_path}ckpts/'
    # os.makedirs(ckpt_save_path, exist_ok=True)
    writer = SummaryWriter('runs')

    fold_num = 5
    data_list = np.load('data/data_list.npy', allow_pickle=True)

    res_overall = {}
    for fold in range(fold_num):
        test_files = data_list[fold]
        train_list = np.delete(data_list, fold, axis=0)
        train_files = [item for row in train_list for item in row]
        logger.info(
            f'fold {fold}: train_num={len(train_files)}, test_num={len(test_files)}'
        )

        train_data, test_data = create_bheh(train_files, test_files)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.num_workers)

        net = model().to(device='cuda')
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        scheduler = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()

        logger.info(
            f'fold    time   | train_l  test_l |    acc     r2 |    '
            f'acc  b_acc      p      r     f1    auc'
        )
        res = []
        for epoch in range(cfg.epoch):
            starttime = time.time()
            # train
            torch.set_grad_enabled(True)
            net.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device='cuda')
                y_batch = y_batch.to(device='cuda')
                y_pred = net(x_batch)
                train_loss = criterion_mse(y_pred, y_batch)
                net.zero_grad()
                train_loss.backward()
                opt.step()
            
            # test
            torch.set_grad_enabled(False)
            net.eval()
            total_loss = []
            true_label_list = []
            pred_label_list = []
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device='cuda')
                y_batch = y_batch.to(device='cuda')
                y_pred = net(x_batch)
                loss_mse = criterion_mse(y_pred, y_batch)
                loss_mae = criterion_mae(y_pred, y_batch)

        # -------------------------------------------------------
        # test
        torch.set_grad_enabled(False)
        net.eval()
        total_sample = 0

        for test_batch_index, (img_batch,
                               label_batch) in enumerate(test_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda().float()
                label_batch = label_batch.cuda().float()

            predict = net(img_batch)
            loss_mse = criterion_mse(predict, label_batch)
            loss_mae = criterion_mae(predict, label_batch)
            total_sample += img_batch.size(0)

        net.train()

        # -------------------------------------------------------
        # logs
        logger.info(f'[{epoch_index+1:03d}/{epochs}] '
                    f'lr:{opt.param_groups[0]["lr"]:.0e} '
                    f'time:{time.time() - startime:.2f}s | '
                    f'train:{loss.item():4f}, '
                    f'mse:{loss_mse.item():4f}, '
                    f'mae:{loss_mae.item():4f}')

        res.append(
            [epoch_index,
             loss.item(),
             loss_mse.item(),
             loss_mae.item()])
        writer.add_scalars(
            timestamp, {
                'train': loss.item(),
                'test_mse': loss_mse.item(),
                'test_mae': loss_mae.item()
            }, epoch_index)

        scheduler.step()

    writer.close()
    save_results(res)

    net.cpu()
    weight_path = f'{ckpt_save_path}{args.model}_f{fold}.pth'
    torch.save(
        {
            'epoch': epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }, weight_path)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    args = make_parser().parse_args()
    cfg = Config()
    timestamp = time.strftime('%y%m%d-%H%M%S', time.localtime())
    output_path = f'results/{timestamp}/'
    os.makedirs(output_path, exist_ok=True)
    logger.add(f'{output_path}/log.txt', format='{message}', level='INFO')

    models = {
        'MyModel': nets.MyModel,
    }
    model = models[args.model]

    logger.info(f'{"=========="*8}\n'
                f'Configuration: -m {args.model} '
                f'bs={cfg.batch_size} ep={cfg.epoch} '
                f'lr={cfg.learning_rate:.0e} step={cfg.gamma}/{cfg.step_size}')

    train(model, cfg, timestamp, output_path)
