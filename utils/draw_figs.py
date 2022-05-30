import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def label_hist():
    path = 'data/annotations/labels.csv'
    labels = pd.read_csv(path)

    # plt.figure()
    # plt.hist(labels['overall']*4+1, bins=5, rwidth=1)
    # plt.hist(labels['overall']*4+1, bins=50, rwidth=1)
    # plt.title('Histogram')
    # plt.show()

    # label_num = np.array(np.unique(labels['overall'], return_counts=True)).T
    # plt.bar(label_num[:,0], label_num[:,1])
    # plt.title('Histogram')
    # plt.show()

    # sns.set()
    # fig, axes = plt.subplots(2, 2)
    # sns.histplot(data=labels['affective']*4+1, bins=5, ax=axes[0,0])
    # plt.xlabel('affective engagement')
    # sns.histplot(data=labels['visual']*4+1, bins=5, ax=axes[0,1])
    # plt.xlabel('visual engagement')
    # sns.histplot(data=labels['behavioral']*4+1, bins=5, ax=axes[1,0])
    # plt.xlabel('behavioral engagement')
    # sns.histplot(data=labels['overall']*4+1, kde=True, bins=5, ax=axes[1,1])
    # plt.xlabel('overall engagement')
    # plt.show()

    path = 'data/annotations/labels_class.csv'
    labels = pd.read_csv(path)

    sns.set()
    fig, axes = plt.subplots(2)
    sns.histplot(data=labels['overall']*4+1, bins=5, ax=axes[0])
    plt.xlabel('overall engagement')
    sns.histplot(data=labels['class'], bins=13, ax=axes[1])
    plt.xlabel('class engagement')
    plt.show()


if __name__ == '__main__':
    label_hist()
