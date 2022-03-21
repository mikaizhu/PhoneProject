import random
import numpy as np
import torch
from pathlib import Path
import logging
import logging.config

import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle 

def save2pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def loadpkl(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


def plot_line_chart(values1, values2, plot_config, mode='acc'):
    # mode: acc or loss
    # 绘制训练集和测试集准确率的折线统计图
    plt.style.use('ggplot')
    values1 = [i.cpu().item() for i in values1] if isinstance(values1[0], torch.Tensor) else values1
    values2 = [i.cpu().item() for i in values2] if isinstance(values2[0], torch.Tensor) else values2
    
    fig = plt.figure()
    plt.ylim([0, (max(max(values1), max(values2)) * 1.1)])
    plt.plot(range(len(values1)), values1, label=plot_config['line'][mode]['labels1'], marker='.')
    plt.plot(range(len(values2)), values2, label=plot_config['line'][mode]['labels2'], marker='.')
    plt.legend(loc='best')
    plt.show()
    plt.savefig(str(Path('figures')/plot_config['exp_name']/plot_config['line'][mode]['filename']))

def plot_bar_group(values1, values2, labels, plot_config):
    # 这里只能展示训练集和测试集两组数据，所以最多两组
    # values1 train ; values2 test ; labels [0, 1, 2...]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=tuple(plot_config['bar']['group']['figsize']))
    rects1 = ax.bar(x - width/2, values1, width, label='train data', alpha=0.5, color='g')
    rects2 = ax.bar(x + width/2, values2, width, label='test data', alpha=0.5, color='b')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(plot_config['bar']['group']['ylabel'])
    ax.set_title(plot_config['bar']['group']['title'])
    ax.set_xticks(x, labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.grid()
    plt.show()
    path = os.path.join('figures', plot_config['exp_name'])
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(str(path) + '/' + plot_config['bar']['group']['filename'])


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


def plot_cm(pred_labels, true_labels, plot_config):
    '''
    label:真实标签，一维ndarray或者数组都行
    preds:模型的预测值
    n_classes:看问题是几分类问题, 默认是10分类问题
    '''
    n_classes = plot_config['num_class']
    cm = confusion_matrix(true_labels, pred_labels)
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
        plt.savefig(str(Path('figures')/plot_config['exp_name']/plot_config['cm']['filename']), dpi=300)
        plt.show()

    plot_Matrix(cm, range(n_classes))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    from read_data import read_data
    from collections import Counter
    import os
    import pickle
    set_seed(42)
    plot_line_chart(plot_line_config)
