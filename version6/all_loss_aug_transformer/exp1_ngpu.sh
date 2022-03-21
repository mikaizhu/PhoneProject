#!/usr/bin/env bash
# /userhome/phone_project/test_v1
#cd $(pwd)
cd /userhome/10ue_xjtu_code/all_loss_aug_transformer
start=$(date +%s)
echo "START: $(date)"
data_path=/userhome/10ue_xjtu_shieldingbox_ChinaMobileFDD_20220118
augmentation_rate=32 # options: 1. 32, 2. 152 3. 272
exp_name="10source_${augmentation_rate}_aug_transformer_abs_centerLoss"
GPU_NUM=2

# -p 参数可以忽略报错，同时可以递归创建文件
mkdir -p log exp/${exp_name} models figures/${exp_name}
random_seed=42
epochs=60
lr=0.0001
balance=False
data_process_method=abs # option: 1. standar  2. abs

cat > log/${exp_name}.log << EOF # 这里会清除文件内容

## 实验方法 ##

模型: Transformer

数据处理方法:
- 先取绝对值
- 然后再最大最小值归一化

EOF

# part1: 训练模型
# 多gpu参考：https://github.com/jia-zhuang/pytorch-multi-gpu-training
# nnodes表示有几个机器，nproc_per_node表示每个机器有几个显卡
/root/miniconda3/envs/py/bin/torchrun --nproc_per_node=${GPU_NUM} main_ngpu.py \
	--data_path ${data_path} \
	--exp_name ${exp_name} \
	--random_seed ${random_seed} \
	--epochs ${epochs} \
	--lr ${lr} \
	--data_process_method ${data_process_method} \
	--augmentation_rate ${augmentation_rate} \
	2>&1 | tee -ai log/${exp_name}.log &
pid=$!
wait ${pid}

echo "END TIME: $(date)"
