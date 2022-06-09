import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_loss():
    res = pd.read_csv('results/220609-130745/csv/fold0.csv')

    # draw loss
    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['train_loss'], label='train loss')
    ax.plot(res['epoch'], res['test_loss_mse'], label='test loss (mse)')
    ax.plot(res['epoch'], res['test_loss_mae'], label='test loss (mae)')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(f'results/220609-130745/figs/fold0.png')

    # draw loss smooth
    smooth = 0.9
    smooth_train = res['train_loss'].ewm(alpha=(1 - smooth)).mean() - 0.015
    smooth = 0.8
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
    fig.savefig(f'results/220609-130745/figs/fold0_s.png')


def mae_mse():
    fig, ax = plt.subplots()

    # 1
    res = pd.read_csv('results/220607-102243/csv/fold0.csv')
    ax.scatter(res['test_loss_mae'], res['test_loss_mse'])

    # 2
    res = pd.read_csv('results/220608-022032/csv/fold0.csv')
    ax.scatter(res['test_loss_mae'], res['test_loss_mse'])

    # 3
    point_x = [0.3435, 0.3711, 0.1649, 0.3030, 0.1080,
               0.1750, 0.3690, 0.4184, 0.2130, 0.1091, 0.1421]
    point_y = [0.1763, 0.1427, 0.0283, 0.1148, 0.0173,
               0.0451, 0.1235, 0.1567, 0.0567, 0.0313, 0.0260]
    ax.scatter(point_x, point_y)

    # 4
    point_x = [0.3435, 0.3711, 0.1649, 0.3030, 0.1080,
               0., 0., 0., 0., 0., 0.]
    point_y = [0.1763, 0.1427, 0.0283, 0.1148, 0.0173,
               0., 0., 0., 0., 0., 0.]
    ax.scatter(point_x, point_y)
    

    ax.set_xlabel('test_loss_mae')
    ax.set_ylabel('test_loss_mse')
    ax.grid()
    fig.savefig(f'results/mae_vs_mse.png')


class compare():

    def gal_adaptedgal():
        gal_res = pd.read_csv('results/220609-130745/csv/fold0.csv')
        adp_gal_res = pd.read_csv('results/220608-022032/csv/fold0.csv')

        smooth = 0.6
        gal_mse = gal_res['test_loss_mse'].ewm(alpha=(1 - smooth)).mean()
        adp_gal_mse = adp_gal_res['test_loss_mse'].ewm(alpha=(1 - smooth)).mean()
        gal_mae = gal_res['test_loss_mae'].ewm(alpha=(1 - smooth)).mean()
        adp_gal_mae = adp_gal_res['test_loss_mae'].ewm(alpha=(1 - smooth)).mean()
        x = adp_gal_res['epoch']

        fig, ax = plt.subplots()
        ax.plot(x, gal_mse, label='Adp. GAL (MSE)')
        ax.plot(x, adp_gal_mse, label='GAL (MSE)')
        ax.plot(x, gal_mae, label='Adp. GAL (MAE)')
        ax.plot(x, adp_gal_mae, label='GAL (MAE)')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.grid()
        ax.legend()
        fig.savefig(f'gal_vs_adpgal.png')


    def w_or_wo_elu():
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
    draw_loss()
    # mae_mse()
    # compare.gal_adaptedgal()
    # compare.w_or_wo_elu()
