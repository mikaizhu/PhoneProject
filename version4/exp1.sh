#!/usr/bin/env bash
# /userhome/phone_project/test_v1
cd /userhome/phone_project/test_v3
#sh ./requirements.sh
start=$(date +%s)
echo "START: $(date)"
data_path='../../16ue_20211223'
exp_name='16source_sort_by_time_balance_abs_minmax_dnn_128aug'

# -p 参数可以忽略报错，同时可以递归创建文件
mkdir -p log exp/${exp_name} models figures/${exp_name}
random_seed=42
epochs=30
lr=0.0001
balance=True
augementation_rate=3
data_process_method=abs # option: 1. standar  2. abs

cat > log/${exp_name}.log << EOF # 这里会清除文件内容

## 实验方法 ##

数据源: 新设备采集的16个源

模型: DNN

数据处理方法:
- 先取绝对值
- 然后再最大最小值归一化

EOF

# part1: 训练模型

python3 -u main.py \
	--data_path ${data_path} \
	--exp_name ${exp_name} \
	--random_seed ${random_seed} \
	--epochs ${epochs} \
	--lr ${lr} \
	--balance \
	--augementation_rate ${augementation_rate} \
	--data_process_method ${data_process_method} \
	2>&1 | tee -ai log/${exp_name}.log &
pid=$!
wait ${pid}

echo "END TIME: $(date)"
