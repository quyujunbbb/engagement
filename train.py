import argparse
import time

import pandas as pd
import matplotlib.pyplot as plt
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
from utils import prepare_data

# torch.manual_seed(0)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', type=str)
    parser.add_argument('-d', '--data', dest='dataset', type=str)

    return parser


def draw_loss(loss_train, loss_test_mse, loss_test_mae, step):
    writer = SummaryWriter('results/runs')
    writer.add_scalars(timestamp, {'loss_train': loss_train, 'loss_test_mse': loss_test_mse, 'loss_test_mae': loss_test_mae}, step)
    writer.close()


def save_results(res):
    res = pd.DataFrame(res, columns=['setp', 'train', 'test_mse', 'test_mae'])
    res.to_csv(f'results/csvs/{timestamp}.csv', index=False)

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


def train_session(model, train_data, test_data, mode, timestamp):
    if mode == 'Reg':
        criterion = nn.MSELoss()
    elif mode == 'Class':
        criterion = nn.CrossEntropyLoss()
    criterion_mae = nn.L1Loss()
    res = []
    writer = SummaryWriter('results/runs')

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

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
        for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda().float()
                label_batch = label_batch.cuda().float()

            # ++++++++++++++++++++++++++++++++++++++++
            predict = net(img_batch)
            loss = criterion(predict, label_batch)
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

        for test_batch_index, (img_batch, label_batch) in enumerate(test_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda().float()
                label_batch = label_batch.cuda().float()

            predict = net(img_batch)
            loss_mse = criterion(predict, label_batch)
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

        res.append([epoch_index, loss.item(), loss_mse.item(), loss_mae.item()])
        writer.add_scalars(timestamp, {'train': loss.item(),
                                       'test_mse': loss_mse.item(),
                                       'test_mae': loss_mae.item()}, epoch_index)

        scheduler.step()

    writer.close()
    save_results(res)

    weight_path = f'checkpoints/{args.model}_{args.dataset}_{timestamp}.pth'
    logger.info(f'Save checkpoints to {weight_path}')
    net.cpu()
    torch.save({'epoch': epochs,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict()
                }, weight_path)


def train_clip(model, data, mode, timestamp):
    if mode == 'Reg':
        criterion = nn.MSELoss()
    elif mode == 'Class':
        criterion = nn.CrossEntropyLoss()
    criterion_mae = nn.L1Loss()
    result_mse, result_mae = {}, {}
    writer = SummaryWriter('results/runs')

    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        logger.info(f'Fold {fold+1}')

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_subsampler)
        test_loader = DataLoader(data, batch_size=batch_size, sampler=test_subsampler)

        net = model()
        if torch.cuda.is_available():
            net = nn.DataParallel(net)
            net.cuda()
        opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
        # opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        scheduler = StepLR(opt, step_size=50, gamma=0.5)

        for epoch_index in range(epochs):
            # ---------------------------------------------------
            # train
            startime = time.time()
            torch.set_grad_enabled(True)
            net.train()
            for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
                # mean = img_batch.mean()
                # std = img_batch.std()
                # print(f'mean = {mean}, std = {std}')

                if torch.cuda.is_available():
                    img_batch = img_batch.cuda().float()
                    label_batch = label_batch.cuda().float()

                predict = net(img_batch)
                loss = criterion(predict, label_batch)

                net.zero_grad()
                loss.backward()
                opt.step()

            logger.info(f'Epoch[{epoch_index+1:03d}/{epochs}] '
                        f'lr:{opt.param_groups[0]["lr"]} '
                        f'time:{time.time() - startime:.2f}s')
            draw_loss('train loss', writer, loss, epoch_index)

            # ---------------------------------------------------
            # test
            torch.set_grad_enabled(False)
            net.eval()
            total_loss_mse = []
            total_loss_mae = []
            total_sample = 0

            for test_batch_index, (img_batch, label_batch) in enumerate(test_loader):
                if torch.cuda.is_available():
                    img_batch = img_batch.cuda().float()
                    label_batch = label_batch.cuda().float()

                predict = net(img_batch)
                loss_mse = criterion(predict, label_batch)
                total_loss_mse.append(loss_mse)
                loss_mae = criterion_mae(predict, label_batch)
                total_loss_mae.append(loss_mae)
                total_sample += img_batch.size(0)

            net.train()

            mean_loss_mse = sum(total_loss_mse) / total_loss_mse.__len__()
            result_mse[fold] = mean_loss_mse
            mean_loss_mae = sum(total_loss_mae) / total_loss_mae.__len__()
            result_mae[fold] = mean_loss_mae

            logger.info(f'(Test) MSE:{mean_loss_mse.item():4f}, '
                        f'MAE:{mean_loss_mae.item():4f}')
            draw_loss('test MSE', writer, mean_loss_mse, epoch_index)
            draw_loss('test MAE', writer, mean_loss_mae, epoch_index)

            scheduler.step()
        break

    writer.close()

    fold_loss = 0.0
    for key, value in result_mse.items():
        logger.info(f'Fold {key}: {value:4f}')
        fold_loss += value
    logger.info(f'Average: {fold_loss/len(result_mse.items()):4f}')

    fold_loss = 0.0
    for key, value in result_mae.items():
        logger.info(f'Fold {key}: {value:4f}')
        fold_loss += value
    logger.info(f'Average: {fold_loss/len(result_mae.items()):4f}')

    weight_path = f'checkpoints/{args.model}_{args.dataset}_{timestamp}.pth'
    logger.info(f'Save checkpoints to {weight_path}')
    net.cpu()
    torch.save({'epoch': epochs,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict()
                }, weight_path)


if __name__ == '__main__':
    args = make_parser().parse_args()
    timestamp = time.strftime('%y%m%d-%H%M%S', time.localtime())
    logger.add(f'results/logs/{timestamp}.txt',
               format='{time:YY-MM-DD HH:mm:ss} | {function}:{line} | {message}',
               level='INFO')

    mode = args.model.split('_')[-1]
    assert(mode in ['Reg', 'Class'])

    models = {
        'Nonlocal_FC3_Reg'   : nets.Nonlocal_FC3_Reg,
        'Nonlocal_FC1_Reg'   : nets.Nonlocal_FC1_Reg,
        'Nonlocal_FC1_Class' : nets.Nonlocal_FC1_Class,
        'FC3_Reg'            : nets.FC3_Reg,
        'FC1_Reg'            : nets.FC1_Reg
    }
    assert(args.model in list(models.keys()))
    model = models[args.model]

    datasets = {
        'roi_s'  : prepare_data.roi_s,
        'roi_l'  : prepare_data.roi_l,
        'roi_ls' : prepare_data.roi_ls,
        'r3d_l'  : prepare_data.r3d_l,
    }
    assert(args.dataset in list(datasets.keys()))
    dataset = datasets[args.dataset]

    epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    step_size = 50
    gamma = 0.1

    logger.info(f'Configuration:'
                f'\n  MODEL         : {model.__name__}'
                f'\n  DATA          : {dataset.__name__}'
                f'\n  BATCH SIZE    : {batch_size}'
                f'\n  EPOCH         : {epochs}'
                f'\n  LEARNING RATE : {learning_rate:.0e}, reduce by {gamma} every {step_size} steps')

    func = args.dataset.split('_')[-1]
    assert(func in ['s', 'l', 'ls'])
    if func == 's' or func == 'ls':
        # session-split
        train_data, test_data = dataset(mode)
        train_session(model, train_data, test_data, mode, timestamp)
    elif func == 'l':
        # clip-split
        data = dataset(mode)
        # mean = data.mean()
        # std = data.std()
        # print(f'mean = {mean}, std = {std}')
        train_clip(model, data, mode, timestamp)
