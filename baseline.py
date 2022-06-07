import os
import time

import numpy as np
import pandas as pd
import torch
from loguru import logger


def create_label(files, labels):
    labels = labels[['session', 'clip', 'overall']]
    label_out = {}

    for file in files:
        session = file.split('.')[0].split('/')[0]
        clip = int(file.split('.')[0].split('/')[1])

        temp = labels[labels['session'] == session]
        label_out[file] = temp[temp['clip'] == clip+1]['overall'].values[0]

    return label_out


def get_ground_truth(train_files, test_files):
    label_path = f'data/annotations/labels_new.csv'
    labels = pd.read_csv(label_path)
    train_label = create_label(train_files, labels)
    test_label = create_label(test_files, labels)

    return train_label, test_label


def save_results(res, fold, output_path):
    os.makedirs(f'{output_path}/csv/', exist_ok=True)
    cols = ['fold', 'mse', 'mae']
    res = pd.DataFrame(res, columns=cols)
    res.to_csv(f'{output_path}/csv/fold{fold}.csv', index=False)


def evaluate(y_true, y_pred, fold, output_path):
    # save csv results
    os.makedirs(f'{output_path}/pred/', exist_ok=True)
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    y_out = pd.DataFrame(columns=['y_true', 'y_pred'])
    y_out['y_true'], y_out['y_pred'] = y_true, y_pred
    y_out.to_csv(f'{output_path}/pred/fold{fold}.csv', index=False)

    # evaluate results
    mse = np.sum((y_true - y_pred)**2) / len(y_true)
    mae = np.sum(np.abs(y_true - y_pred)) / len(y_true)

    return mse, mae


def train(output_path):
    fold_num = 5
    data_list = np.load('data/data_list.npy', allow_pickle=True)

    res_overall = {}
    for fold in range(fold_num):
        test_files = data_list[fold]
        train_list = np.delete(data_list, fold, axis=0)
        train_files = [item for row in train_list for item in row]

        train_label, test_label = get_ground_truth(train_files, test_files)

        y_true = np.fromiter(test_label.values(), dtype=float)
        # Op1: random guess
        # y_pred = np.random.rand(len(y_true))
        # Op2: average guess
        y_train = np.fromiter(train_label.values(), dtype=float)
        y_pred = np.full(len(y_true), np.average(y_train))

        res = []
        mse, mae = evaluate(y_true, y_pred, fold, output_path)
            
        logger.info(f'fold{fold}  {mse.item(): 2.4f}  {mae.item(): 2.4f}')
        res.append([fold, mse, mae])
        res_overall[fold] = [mse, mae]
        save_results(res, fold, output_path)

    avg_test_loss_mse, avg_test_loss_mae = 0.0, 0.0
    for _, value in res_overall.items():
        avg_test_loss_mse += value[0]
        avg_test_loss_mae += value[1]
    avg_test_loss_mse = avg_test_loss_mse / fold_num
    avg_test_loss_mae = avg_test_loss_mae / fold_num
    logger.info(f'---------------------\n       '
                f'{avg_test_loss_mse:.4f}  {avg_test_loss_mae:.4f}')


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    timestamp = time.strftime('%y%m%d-%H%M%S', time.localtime())
    output_path = f'results/{timestamp}/'
    os.makedirs(output_path, exist_ok=True)

    logger.add(f'{output_path}/log.txt', format='{message}', level='INFO')
    logger.info(f'Configuration: average_guess')

    train(output_path)
