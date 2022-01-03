#!/home/zwl/miniconda3/envs/asr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
from data import Data, MyDataLoader, MyDataset
from trainer import Trainer
import utils
from utils import set_seed
from pathlib import Path
import os
from read_data import read_data
import gc
import numpy as np
#import toml
import json
from utils import save2pkl
import argparse

from model import DnnModel

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--exp_name', type=str)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--augementation_rate', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--data_process_method', type=str, default='standar')
args = parser.parse_args() # 解析参数

# 获取打印参数
#plot_config = toml.load('./plot_config.toml')
with open('plot_config.json', 'rb') as f:
    plot_config = json.load(f)
plot_config['exp_name'] = args.exp_name

# 打印出所有参数
for arg in vars(args):
    print("{:<20}{:<}".format(arg, str(getattr(args, arg))))

set_seed(args.random_seed)

train_config = {
    'batch_size':args.batch_size,
    'shuffle':True,
    'drop_last':True,
    'pin_memory':True,
}

test_config = {
    'batch_size':args.batch_size,
    'shuffle':True,
    'drop_last':True,
    'pin_memory':True,
}
print('Stage1: data load')
data = Data()
phone_data = read_data()
# 获取数据分布图
x_train, x_test, y_train, y_test = phone_data.read_data_by_sort_time(args.data_path,
        balance=args.balance) 

plot_config['num_class'] = phone_data.num_class

save2pkl(phone_data.id2phone, 'exp/' + args.exp_name + '/id2phone.pkl') # 保存标签
utils.plot_bar_group(values1=phone_data.trainBincount, values2=phone_data.testBincount,
        labels=phone_data.true_label, plot_config=plot_config)

gc.collect()
print('load data successful')
x_train = data.process(x_train, method=args.data_process_method)
x_test = data.process(x_test, method=args.data_process_method)
print('data process done')

gc.collect()
train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

train_loader = MyDataLoader(train_dataset, **train_config)
test_loader = MyDataLoader(test_dataset, **test_config)

window_len = x_train.shape[1]-args.augementation_rate
model = DnnModel(window_len, phone_data.num_class)

epochs = args.epochs
lr = args.lr
criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'model' : model,
    'exp':args.exp_name,
    'optimizer' : optimizer,
    'scheduler' : scheduler,
    'criterion' : criterion,
    'epochs' : epochs,
    'train_loader' : train_loader,
    'test_loader' : test_loader,
    'model_save_name' : args.exp_name + '.model',
    'augementation_rate' : args.augementation_rate,
    'window_len': window_len,
}

print('Stage2: model training')
trainer = Trainer(**config)
trainer.train()

# 模型训练完毕， 开始画图
path = Path('exp')/args.exp_name
train_acc, test_acc = utils.loadpkl(str(path/'train_acc.pkl')), utils.loadpkl(str(path/'test_acc.pkl'))
train_loss, test_loss = utils.loadpkl(str(path/'train_loss.pkl')), utils.loadpkl(str(path/'test_loss.pkl'))
# 准确率图
utils.plot_line_chart(train_acc, test_acc, plot_config, mode='acc')
# loss图
utils.plot_line_chart(train_loss, test_loss, plot_config, mode='loss')
# 混淆矩阵图
pred_labels = utils.loadpkl(str(Path('exp')/args.exp_name/'pre_labels.pkl'))
true_labels = utils.loadpkl(str(Path('exp')/args.exp_name/'true_labels.pkl'))
utils.plot_cm(pred_labels, true_labels, plot_config) # preds, label
