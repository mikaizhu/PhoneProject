#!/home/zwl/miniconda3/envs/asr/bin/python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json

from center_loss import CenterLoss

# multi gpu
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from data import Data, MyDataLoader, MyDataset
from trainer import Trainer

import utils
from utils import set_seed

from pathlib import Path

from read_data import read_data

import gc
import numpy as np

from utils import save2pkl

import argparse

from model import DnnModel
from model import TransfromerModel

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--exp_name', type=str)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--augmentation_rate', type=int, default=0)
parser.add_argument('--data_process_method', type=str, default='standar')
args = parser.parse_args() # 解析参数

# 多gpu设置

n_gpus = torch.cuda.device_count()
world_size = n_gpus # 全局进程个数，一般为gpu个数
assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
torch.distributed.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
args.local_rank = local_rank

# 打印出所有参数

if args.local_rank == 0:
    for arg in vars(args):
        print("{:<20}{:<}".format(arg, str(getattr(args, arg))))

# 获取打印参数
with open('plot_config.json', 'rb') as f:
    plot_config = json.load(f)
plot_config['exp_name'] = args.exp_name

set_seed(args.random_seed)

train_config = {
    'batch_size':args.batch_size,
    'drop_last':True,
    'pin_memory':True,
}

test_config = {
    'batch_size':args.batch_size,
    'shuffle':True,
    'drop_last':True,
    'pin_memory':True,
}

if args.local_rank == 0:
    print('Stage1: data load')
data = Data()
phone_data = read_data()
# 获取数据分布图
x_train, x_test, y_train, y_test = phone_data.read_data_by_round(args.data_path)
plot_config['num_class'] = phone_data.num_class

save2pkl(phone_data.phone2id, 'exp/' + args.exp_name + '/phone2id.pkl') # 保存标签
utils.plot_bar_group(values1=phone_data.trainBincount, values2=phone_data.testBincount,
        labels=phone_data.true_label, plot_config=plot_config)

gc.collect()
if args.local_rank == 0:
    print('load data successful')
x_train = data.process(x_train, method=args.data_process_method)
x_test = data.process(x_test, method=args.data_process_method)
if args.local_rank == 0:
    print('data process done')

gc.collect()
train_dataset = MyDataset(x_train, y_train)
train_sampler = DistributedSampler(train_dataset)

test_dataset = MyDataset(x_test, y_test)

#train_loader = MyDataLoader(train_dataset, **train_config)
train_loader = MyDataLoader(train_dataset, sampler=train_sampler, **train_config)
test_loader = MyDataLoader(test_dataset, **test_config)

args.window_len = x_train.shape[1]-args.augmentation_rate
# aug transformer
if args.augmentation_rate:
    args.d_model = 60
    args.n_head = 6
    args.embed_dim = args.window_len / 2 / args.d_model
else:
    args.d_model = 64
    args.n_head = 8
    args.embed_dim = args.window_len / 2 / args.d_model
assert args.embed_dim % 1 == 0, f"args.embed_dim 必须为整数"
args.embed_dim = int(args.embed_dim)
#model = DnnModel(window_len, phone_data.num_class).to(device)
model = TransfromerModel(L1=args.window_len, num_class=phone_data.num_class, d_model=args.d_model, n_head=args.n_head, dropout=0.2).to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

epochs = args.epochs
lr = args.lr

criterion_xent = nn.CrossEntropyLoss()
criterion_cent = CenterLoss(num_classes=phone_data.num_class, use_gpu=True, feat_dim=512)
optimizer_cent = optim.AdamW(criterion_cent.parameters(), lr=0.01)
optimizer_model = optim.AdamW(model.parameters(), lr=args.lr)
#criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)
#optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, mode='min', factor=0.1, patience=5)


config = {
    'device': device,
    'model' : model,
    'exp':args.exp_name,
    #'optimizer' : optimizer,
    'criterion_cent':criterion_cent,
    'criterion_xent':criterion_xent,
    'optimizer_model':optimizer_model,
    'optimizer_cent':optimizer_cent,
    'scheduler' : scheduler,
    #'criterion' : criterion,
    'epochs' : epochs,
    'train_loader' : train_loader,
    'test_loader' : test_loader,
    'model_save_name' : args.exp_name + '.model',
    'args': args,
    'augmentation_rate' : args.augmentation_rate,
    'window_len': args.window_len,
}

if args.local_rank == 0:
    print('Stage2: model training')
trainer = Trainer(**config)
trainer.train()

# 模型训练完毕， 开始画图
if args.local_rank == 0:
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
