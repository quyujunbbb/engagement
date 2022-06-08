import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_loss():
    res = pd.read_csv('results/220607-102243/csv/fold0.csv')

    # draw loss
    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['train_loss'], label='train loss')
    ax.plot(res['epoch'], res['test_loss_mse'], label='test loss (mse)')
    ax.plot(res['epoch'], res['test_loss_mae'], label='test loss (mae)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(f'results/220607-102243/figs/fold0.png')

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
    fig.savefig(f'results/220607-102243/figs/fold0_s.png')

    # draw mae vs. mse
    fig, ax = plt.subplots()
    ax.plot(res['test_loss_mae'], res['test_loss_mse'])
    ax.set_xlabel('test_loss_mae')
    ax.set_ylabel('test_loss_mse')
    ax.grid()
    fig.savefig(f'results/220607-102243/figs/fold0_mae_vs_mse.png')


def compare_gal_adaptedgal():
    gal_res = pd.read_csv('results/220607-082816/csv/fold0.csv')
    adp_gal_res = pd.read_csv('results/220607-160238/csv/fold0.csv')

    smooth = 0.6
    gal_mse = gal_res['test_loss_mse'].ewm(alpha=(1 - smooth)).mean()
    gal_mae = gal_res['test_loss_mae'].ewm(alpha=(1 - smooth)).mean()
    adp_gal_mse = adp_gal_res['test_loss_mse'].ewm(alpha=(1 - smooth)).mean()
    adp_gal_mae = adp_gal_res['test_loss_mae'].ewm(alpha=(1 - smooth)).mean()
    x = adp_gal_res['epoch']

    fig, ax = plt.subplots()
    ax.plot(x, gal_mse, label='GAL (MSE)')
    ax.plot(x, gal_mae, label='GAL (MAE)')
    ax.plot(x, adp_gal_mse, label='Adp. GAL (MSE)')
    ax.plot(x, adp_gal_mae, label='Adp. GAL (MAE)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(f'gal_vs_adpgal.png')


def compare_w_or_wo_elu():
    adp_gal_res = pd.read_csv('results/220607-160238/csv/fold0.csv').iloc[:20,:]
    adp_gal_elu_res = pd.read_csv('results/220608-022032/csv/fold0.csv')

    smooth = 0.6
    adp_gal_elu_mse = adp_gal_elu_res['test_loss_mse'].ewm(alpha=(1 - smooth)).mean()
    adp_gal_elu_mae = adp_gal_elu_res['test_loss_mae'].ewm(alpha=(1 - smooth)).mean()
    adp_gal_mse = adp_gal_res['test_loss_mse'].ewm(alpha=(1 - smooth)).mean()
    adp_gal_mae = adp_gal_res['test_loss_mae'].ewm(alpha=(1 - smooth)).mean()
    x = adp_gal_res['epoch']

    fig, ax = plt.subplots()
    ax.plot(x, adp_gal_elu_mse, label='Adp. GAL + elu (MSE)')
    ax.plot(x, adp_gal_elu_mae, label='Adp. GAL + elu (MAE)')
    ax.plot(x, adp_gal_mse, label='Adp. GAL (MSE)')
    ax.plot(x, adp_gal_mae, label='Adp. GAL (MAE)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(f'w_or_wo_elu.png')


if __name__ == '__main__':
    # draw_loss()
    compare_gal_adaptedgal()
    compare_w_or_wo_elu()
