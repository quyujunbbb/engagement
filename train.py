import argparse
import os
import time

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

from models import nets_old
from utils import prepare_data


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


def train(model, timestamp, output_path):
    ckpt_save_path = f'{output_path}ckpts/'
    os.makedirs(ckpt_save_path, exist_ok=True)
    writer = SummaryWriter('runs')

    fold_num = 5
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    res = []

    train_data, test_data = dataset()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)

    net = model()
    if torch.cuda.is_available():
        net = nn.DataParallel(net)
        net.cuda()
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(opt, step_size=step_size, gamma=gamma)
    # ++++++++++++++++++++++++++++++++++++++++
    # scaler = amp.GradScaler()
    # ++++++++++++++++++++++++++++++++++++++++

    for epoch_index in range(epochs):
        # -------------------------------------------------------
        # train
        startime = time.time()
        torch.set_grad_enabled(True)
        net.train()
        for train_batch_index, (img_batch,
                                label_batch) in enumerate(train_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda().float()
                label_batch = label_batch.cuda().float()

            # ++++++++++++++++++++++++++++++++++++++++
            predict = net(img_batch)
            loss = criterion_mse(predict, label_batch)
            # if using amp, replace previous two lines with followings
            # with amp.autocast():
            #     predict = net(img_batch)
            #     loss = criterion(predict, label_batch)
            # ++++++++++++++++++++++++++++++++++++++++

            net.zero_grad()
            # ++++++++++++++++++++++++++++++++++++++++
            loss.backward()
            opt.step()
            # if using amp, replace previous two lines with followings
            # scaler.scale(loss).backward()
            # scaler.step(opt)
            # scaler.update()
            # ++++++++++++++++++++++++++++++++++++++++

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
        'Nonlocal_FC3_Reg'   : nets_old.Nonlocal_FC3_Reg,
        'Nonlocal_FC1_Reg'   : nets_old.Nonlocal_FC1_Reg,
        'Nonlocal_FC1_Class' : nets_old.Nonlocal_FC1_Class,
        'FC3_Reg'            : nets_old.FC3_Reg,
        'FC1_Reg'            : nets_old.FC1_Reg
    }
    assert (args.model in list(models.keys()))
    model = models[args.model]

    datasets = {
        'roi_ls': prepare_data.roi_ls,
    }
    assert (args.dataset in list(datasets.keys()))
    dataset = datasets[args.dataset]

    logger.info(
        f'{"=========="*8}\n'
        f'Configuration: -m {args.model} -d {args.dataset} '
        f'bs={cfg.batch_size} ep={cfg.epoch} '
        f'lr={cfg.learning_rate:.0e} step={cfg.gamma}/{cfg.step_size}'
    )

    # train(model, timestamp, output_path)
