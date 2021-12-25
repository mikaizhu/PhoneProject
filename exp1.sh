#!/usr/bin/env bash
# 实验1：
# data: 16 手机源
# data process method:

cd /userhome/phone_project

data_path='../16ue_20211223'
exp_name='16_source_sort_by_time'
random_seed=42
epochs=30
lr=0.0001

######

python3 main.py \
	--data_path ${data_path} \
	--exp_name ${exp_name} \
	--random_seed ${random_seed} \
	--epochs ${epochs} \
	--lr ${lr} &
pid=$!
wait ${pid}

# part1: 训练模型

# part2: 绘制数据统计图

# part3: 绘制train, test loss accuracy 曲线图

# part4: 绘制混淆矩阵图
