import random
import numpy as np
import torch
from pathlib import Path
import logging
import logging.config

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_bar_group(plot_config):
    # 这里只能展示训练集和测试集两组数据，所以最多两组
    labels = plot_config['labels']
    values1 = plot_config['values1'] # train
    values2 = plot_config['values2'] # test

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=plot_config['figsize'])
    rects1 = ax.bar(x - width/2, values1, width, label='train data', alpha=0.5, color='g')
    rects2 = ax.bar(x + width/2, values2, width, label='test data', alpha=0.5, color='b')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(plot_config['ylabel'])
    ax.set_title(plot_config['title'])
    ax.set_xticks(x, labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.grid()
    plt.show()
    plt.savefig('./figures/'+plot_config['filename'])


def plot_bar(plot_config):
    plt.figure(figsize=plot_config['figsize'])
    mean_values = plot_config['values']
    bar_labels = plot_config['labels']
    # plot bars
    x_pos = list(range(len(bar_labels)))
    rects = plt.bar(x_pos, mean_values, align='center', alpha=0.5)
    
    # label bars
    def autolabel(rects):
        for ii,rect in enumerate(rects):
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%s'% (mean_values[ii]),
                ha='center', va='bottom')
    autolabel(rects)
    
    # set height of the y-axis
    #max_y = max(zip(mean_values, variance)) # returns a tuple, here: (3, 5)
    plt.ylim([0, (max(mean_values) * 1.1)])
    # set axes labels and title
    plt.ylabel(plot_config['ylabel'])
    plt.xlabel(plot_config['xlabel'])
    plt.xticks(x_pos, bar_labels)
    plt.title(plot_config['title'])
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.grid()
    plt.savefig('./figures/'+plot_config['filename'])
    plt.show()


def plot_cm(preds, label, file_name, n_classes=10):
    '''
    label:真实标签，一维ndarray或者数组都行
    preds:模型的预测值
    n_classes:看问题是几分类问题, 默认是10分类问题
    '''
    cm = confusion_matrix(label, preds)
    def plot_Matrix(cm, classes, title=None,  cmap=plt.cm.Blues):
        plt.rc('font',family='Times New Roman',size='8')   # 设置字体样式、大小

        # 按行进行归一化
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if int(cm[i, j]*100 + 0.5) == 0:
                    cm[i, j]=0

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='Actual',
            xlabel='Predicted')

        # 通过绘制格网，模拟每个单元格的边框
        ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # 将x轴上的lables旋转45度
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # 标注百分比信息
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if int(cm[i, j]*100 + 0.5) > 0:
                    ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                            ha="center", va="center",
                            color="white"  if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(file_name, dpi=300)
        plt.show()

    plot_Matrix(cm, range(n_classes))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_logging_config(file_name='trian.log', preserve=True):
    # 默认创建log文件夹，存放所有日志文件
    log_ = Path('./log')
    file_path = log_/Path(file_name)
    if not log_.exists():
        log_.mkdir()
    if not preserve:
    #else:
        # 删除日志原有的内容
        with file_path.open('w') as f:
            f.truncate()

    file_name = str(file_path)

    d = {
        'version':1,
        'formatters':{
            'message':{
                'format':'%(message)s',
            },
            'time_message':{
                'format':'%(asctime)s %(message)s',
                'datefmt':'%Y/%m/%d %H:%M:%S',
            }
        },
        'handlers':{
            'console':{
                'class':'logging.StreamHandler',
                'level':'INFO',
                'formatter':'message'
            },
            'file':{
                'class':'logging.FileHandler',
                'filename':file_name,
                'level':'INFO',
                'formatter':'time_message'
            },
        },
        'loggers':{
            'logger':{
                'level':'INFO',
                'handlers':['file', 'console'],
            },
        },
    }

    return d

if __name__ == '__main__':
    from read_data import read_data
    from collections import Counter
    import os

    read_data = read_data()
    x_train, x_test, y_train, y_test = read_data.read_data_by_sort_time('../16ue_20211223/')

    set_seed(42)
    plot_config = {
            'figsize':(10, 5),
            #'values':list(Counter(y_test).values()),
            'values1':list(Counter(y_train).values()), # values1放置train data
            'values2':list(Counter(y_test).values()), # values2放置test data
            'labels':list(Counter(y_test).keys()),
            'xlabel':'labels',
            'ylabel':'number of symbols',
            'title':'number of train and test symbols',
            'filename':'16source_train_test_group_time_rank_symbols_bar.png',
            }
    plot_bar_group(plot_config)
