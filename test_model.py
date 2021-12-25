#!/home/zwl/miniconda3/envs/asr/bin/python3
import gc
import logging
import logging.config
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from read_data import read_data
import data
from data import Data, MyDataLoader, MyDataset
from utils import get_logging_config
from utils import plot_cm
sys.path.append('..')


# logger set
logging.config.dictConfig(get_logging_config(file_name='test.log'))
logger = logging.getLogger('logger')

# 数据读取
train_config = {
    'batch_size':128,
    'shuffle':True,
    'drop_last':True,
    'pin_memory':True,
}

test_config = {
    'batch_size':128,
    'shuffle':True,
    'drop_last':True,
    'pin_memory':True,
}

print('Stage1: data load')
data = Data(logger)
read_data = read_data()
x_train, x_test, y_train, y_test = read_data.read_data_from_16_source()
#un_label = [0, 1, 2, 3, 4, 5, 6, 7] # 设置未知源

#for label in un_label:
#    idx = (y_test == label)
#    x_test = np.delete(x_test, idx, axis=0)
#    y_test = np.delete(y_test, idx)
#def get_label_map(label):
#    true_label = label
#    label = set(label)
#    label_map = dict(zip(label, range(len(label))))
#    true_label = list(map(lambda x:label_map.get(x), true_label))
#    return np.array(true_label)
#
#y_test = get_label_map(y_test) if un_label else y_test # 如果有未知源，就标签映射

class_num = len(np.bincount(y_test))
gc.collect()
print('load data successful')
x_test = data.process(x_test)
print('process data successful')

gc.collect()
test_dataset = MyDataset(x_test, y_test)

test_loader = MyDataLoader(test_dataset, **test_config)

del x_train, y_train

gc.collect()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).long()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = resnet18()
model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.maxpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
model.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)
# 数据测试
model.load_state_dict(torch.load('./23_classification.model'))
model = model.to(device)

true_label = []
pre_label = []
with torch.no_grad():
    model.eval()
    acc_num = 0
    for feature, label in test_loader:
        feature = feature.reshape(-1, 2, 64, 64).to(device)
        label = label.to(device)
        pre = model(feature).argmax(1)
        acc_num += (pre == label).sum()

        true_label.extend(label.cpu().tolist())
        pre_label.extend(pre.cpu().tolist())
logger.info(acc_num / len(y_test))
plot_cm(pre_label, true_label, '23_classification', n_classes=23)
