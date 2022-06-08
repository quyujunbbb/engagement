import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models import nets
from utils.bheh import create_bheh


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', type=str)
    parser.add_argument('-l', '--layer', dest='layer', type=str)

    return parser


class Config:
    ep = 60
    bs = 16
    lr = 1e-4
    step = 20
    gamma = 0.1
    worker = 4


def save_results(res, fold, output_path):
    # save results
    os.makedirs(f'{output_path}/csv/', exist_ok=True)
    cols = ['epoch', 'train_loss', 'test_loss_mse', 'test_loss_mae']
    res = pd.DataFrame(res, columns=cols)
    res.to_csv(f'{output_path}/csv/fold{fold}.csv', index=False)

    # draw loss
    os.makedirs(f'{output_path}/figs/', exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['train_loss'], label='train loss')
    ax.plot(res['epoch'], res['test_loss_mse'], label='test loss (mse)')
    ax.plot(res['epoch'], res['test_loss_mae'], label='test loss (mae)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(f'{output_path}/figs/fold{fold}.png')

    # draw loss smooth
    smooth = 0.8
    smooth_train = res['train_loss'].ewm(alpha=(1 - smooth)).mean()
    smooth_test_mse = res['test_loss_mse'].ewm(alpha=(1 - smooth)).mean()
    smooth_test_mae = res['test_loss_mae'].ewm(alpha=(1 - smooth)).mean()

    fig, ax = plt.subplots()
    ax.plot(res['epoch'], smooth_train, label='train loss')
    ax.plot(res['epoch'], smooth_test_mse, label='test loss (mse)')
    ax.plot(res['epoch'], smooth_test_mae, label='test loss (mae)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(f'{output_path}/figs/fold{fold}_s.png')


def train(model, layer, cfg, timestamp, output_path):
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

        train_data, test_data = create_bheh(train_files, test_files)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=cfg.bs,
                                  shuffle=True,
                                  num_workers=cfg.worker)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=cfg.bs,
                                 shuffle=True,
                                 num_workers=cfg.worker)

        if args.model == 'NonLocal':
            net = model().to(device='cuda')
        else:
            net = model(input_size=1024,
                        output_size=64,
                        head_num=3,
                        layer=layer).to(device='cuda')
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        scheduler = StepLR(opt, step_size=cfg.step, gamma=cfg.gamma)
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()

        logger.info(f'fold {fold+1}')
        res = []
        for epoch in range(cfg.ep):
            starttime = time.time()

            # train
            torch.set_grad_enabled(True)
            net.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device='cuda')
                y_batch = y_batch.clone().detach().to(torch.float).view(
                    -1, 1).to(device='cuda')
                y_pred = net(x_batch)

                train_loss = criterion_mse(y_pred, y_batch)
                net.zero_grad()
                train_loss.backward()
                opt.step()

            # test
            torch.set_grad_enabled(False)
            net.eval()
            total_loss_mse, total_loss_mae = [], []
            y_true_list, y_pred_list = [], []
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device='cuda')
                y_batch = y_batch.clone().detach().to(torch.float).view(
                    -1, 1).to(device='cuda')
                y_pred = net(x_batch)

                loss_mse = criterion_mse(y_pred, y_batch)
                loss_mae = criterion_mae(y_pred, y_batch)
                total_loss_mse.append(loss_mse)
                total_loss_mae.append(loss_mae)

                y_true_list.append(y_batch.cpu().detach().numpy())
                y_pred_list.append(y_pred.cpu().detach().numpy())

            mean_loss_mse = sum(total_loss_mse) / total_loss_mse.__len__()
            mean_loss_mae = sum(total_loss_mae) / total_loss_mae.__len__()

            # logs
            logger.info(
                f'ep{epoch+1:02d} {time.time()-starttime:.1f} '
                f'{train_loss.item():.4f} '
                f'{mean_loss_mse.item():.4f} {mean_loss_mae.item():.4f}')
            writer.add_scalars(
                f'{timestamp}/{fold}', {
                    'train loss': train_loss.item(),
                    'test loss (mse)': mean_loss_mse.item(),
                    'test loss (mae)': mean_loss_mae.item()
                }, epoch)
            res.append([
                epoch,
                train_loss.item(),
                mean_loss_mse.item(),
                mean_loss_mae.item()
            ])
            res_overall[fold] = [
                train_loss.item(),
                mean_loss_mse.item(),
                mean_loss_mae.item()
            ]

            net.train()
            scheduler.step()

        writer.close()
        save_results(res, fold, output_path)

    avg_train_loss, avg_test_loss_mse, avg_test_loss_mae = 0.0, 0.0, 0.0
    for _, value in res_overall.items():
        avg_train_loss += value[0]
        avg_test_loss_mse += value[1]
        avg_test_loss_mae += value[2]
    avg_train_loss = avg_train_loss / fold_num
    avg_test_loss_mse = avg_test_loss_mse / fold_num
    avg_test_loss_mae = avg_test_loss_mae / fold_num
    logger.info(f'-------------------------------\n'
                f'           {avg_train_loss:.4f} '
                f'{avg_test_loss_mse:.4f} {avg_test_loss_mae:.4f}')


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
        'NonLocal'   : nets.NonLocal,
        'GAT'        : nets.GAT,
        'NonLocalGAT': nets.NonLocalGAT,
    }
    model = models[args.model]

    logger.info(f'Configuration: {args.model}-{args.layer} bs={cfg.bs} ep={cfg.ep} '
                f'lr={cfg.lr:.0e} step={cfg.gamma}/{cfg.step}')

    train(model, args.layer, cfg, timestamp, output_path)
