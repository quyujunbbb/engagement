import argparse
import time

import torch
from loguru import logger
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from models import nets
from utils import prepare_data


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--data', dest='dataset', type=str)

    return parser


def eval(net, data):
    if mode == 'Reg':
        criterion = nn.MSELoss()
    elif mode == 'Class':
        criterion = nn.CrossEntropyLoss()
    criterion_mae = nn.L1Loss()

    test_loader = DataLoader(data, batch_size=batch_size)

    startime = time.time()
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

    mean_loss_mse = sum(total_loss_mse) / total_loss_mse.__len__()
    mean_loss_mae = sum(total_loss_mae) / total_loss_mae.__len__()

    logger.info(f'(Test) MSE:{mean_loss_mse.item():4f}, '
                f'MAE:{mean_loss_mae.item():4f}')


if __name__ == "__main__":
    pretrained_path = 'checkpoints/Nonlocal_FC3_Reg_roi_l_220306-162757.pth'
    timestamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
    logger.add(f'logs/test_{timestamp}.log',
               format="{time:YY-MM-DD HH:mm:ss} | {level:5s} | {function}:{line} | {message}",
               level="INFO")
    args = make_parser().parse_args()

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
        'roi_s' : prepare_data.roi_s,
        'roi_l' : prepare_data.roi_l,
        'r3d_l' : prepare_data.r3d_l
    }
    assert(args.dataset in list(datasets.keys()))
    dataset = datasets[args.dataset]

    epochs = 60
    batch_size = 16
    learning_rate = 1e-3

    logger.info(f'Configuration:'
                f'\n    :: MODEL         - {model.__name__}'
                f'\n    :: DATASET       - {dataset.__name__}'
                f'\n    :: EPOCH         - {epochs}'
                f'\n    :: BATCH SIZE    - {batch_size}'
                f'\n    :: LEARNING RATE - {learning_rate}')

    net = model()
    if torch.cuda.is_available():
        net = nn.DataParallel(net)
        net.cuda()
    net.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
    net.eval()

    test_data = dataset(mode)
    test_data.cuda()

    eval(test_data)
