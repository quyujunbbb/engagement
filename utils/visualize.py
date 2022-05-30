import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def label_hist():
    path = 'data/annotations/labels.csv'
    labels = pd.read_csv(path)

    path = 'data/annotations/labels.csv'
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
