#!/usr/bin/env bash
# /userhome/phone_project/test_v1
cd $(pwd)
start=$(date +%s)
echo "START: $(date)"
data_path='../../16ue_20211223'
exp_name='16_source_sort_by_time_unbalance_abs_minmax'

# -p 参数可以忽略报错，同时可以递归创建文件
mkdir -p log exp/${exp_name} models figures/${exp_name}
random_seed=42
epochs=2
lr=0.0001
balance=False
data_process_method=abs # option: 1. standar  2. abs

cat > log/${exp_name}.log << EOF # 这里会清除文件内容

## 实验方法 ##

数据源: 新设备采集的16个源

数据处理方法:
- 先取绝对值
- 然后再最大最小值归一化

EOF

# part1: 训练模型

/root/miniconda3/envs/py/bin/python3 -u main.py \
	--data_path ${data_path} \
	--exp_name ${exp_name} \
	--random_seed ${random_seed} \
	--epochs ${epochs} \
	--lr ${lr} \
	--data_process_method ${data_process_method} \
	2>&1 | tee -ai log/${exp_name}.log &
pid=$!
wait ${pid}

echo "END: $(date)"
