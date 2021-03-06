# 环境说明

OS: Ubuntu20.04

requirements:
- pytorch==1.10.1(GPU support)
- torchvision==0.11.2
- toml

# 文件说明

main.py: 整个程序的运行入口

data.py: 数据预处理

read_data.py: 数据读取方法

trainer.py: 模型训练方法

utils.py: 画图等工具

test_model.py: 保存模型后，用来测试模型，并画图

> 这里保存完模型后，再调用模型画图，效果比较好

子目录文件说明:

- exp: 存放准确率和loss的文件，方便画图，每个实验对应一个文件夹

- models: 存放训练的模型

- figures: 存放生成的图片，每个实验对应一个文件夹

- log: 存放所有实验的运行日志，报错信息等
# version1实验

目前只使用了resnet模型，后续在version1的基础上修改代码更新version2模型

运行前先配置好文件：
- exp*.sh

运行脚本：
```
sh exp1.sh
```

# version2实验

将模型换成DNN，这样可以更方便使用自己定义的模型，只要定义在model.py文件中即可

# version3实验

增加了多gpu训练模型，主要文件有：
- exp1_ngpu.sh
- main_ngpu.py
- trainer.py

参考：https://github.com/jia-zhuang/pytorch-multi-gpu-training

主要思路：

> 1. 使用distributed data parallel多进程进行单机多卡实验，先前版本是使用命令行
> `python -m torch.distributed.lunch`, 现在更换为torchrun即可, 参考
> exp1_ngpu.sh
> 2. 多进程会连续打印问题：使用if args.local_rank==0方法来解决

# version4

实现了切片数据增强

# version5

新增功能：
- 这个版本使用的是2个gpu，1个gpu不能运行
- 同时修改了transformer的网络结构，使得准确率提高

# version6
新增功能：
- 使用两个gpu才能运行
- 在version5的版本下，将损失函数改成了cross entropy + center
  loss，使得准确率进一步提升

# 程序设计思路

整个程序分为下面几个模块(每个模块应该做到方便单独调试)：

1. 数据读取模块
  - [x] 方便以后添加多种数据读取方式
  - [x] 包含多种数据读取方式, 实验不同，数据读取的方式也不同
  - [x] 方便模块单独调试

2. 数据处理模块
  - [x] 方便以后添加多种数据处理方式
  - [x] 包换多种数据处理方式
  - [x] 方便模块单独调试

3. 模型训练模块
  - [x] 方便添加多种训练方式，如数据增强
  - [x] 方便模块单独调试

4. 画图模块
  - [x] 方便添加多种画图方式
  - [x] 方便模块单独调试

5. 模型测试模块

文件保存名字命名规则：(基础名字exp_name)
- 模型命名规则：exp_name + .model
- 数据命名规则：exp_name + _trainLoss.pkl; exp_name + _testLoss.pkl; exp_name +
  _trainAcc.pkl...
- 图片命名规则：exp_name + LineChart.png; exp_name + cm.png

文件保存规则：
- 数据保存：应该有下一级别文件，下一级别子目录名字为实验名
- 图片保存：和数据保存方式一样，都应该有下一级子目录
- 模型保存：直接保存，没有子文件目录

# TODO

- [x] 支持多GPU训练
- [ ] 添加checkpoint，程序意外中断也可以从checkpoint位置开始训练


# 问题记录及解决办法

1. 记录下所有的打印信息，包括报错的内容


>解决方法：
>
>linux环境下可以编写shell脚本来运行python代码，所以可以使用标准输出，
>来记录日志，这样python文件中只要使用print函数即可, 但遇到的问题还有一下几点;
>1. 如何不清空文件内容
>2. 如何让记录及时输出
>3. 如何记录错误信息
>4. 如何不让control + c中断输出
>
>对于1；使用tee命令的-a参数，为append到后面.
>
>对于2; print输出先是输出到缓存区中，只有在程序运行完后，才会通过Linux管道命令
>进行记录，从而不能及时记录信息，使用python -u test.py参数执行代码, 可以让缓存
>区的内容及时输出.
>
>对于3；使用Linux中的`2>&1`方法, 2表示stderr 1表示stdout，Linux中所有东西都是文
>件，2 和 1 也不例外，上面命令表示把错误也写入到标准输出中
>
>对于4；使用tee的-i命令可以忽略中断，这样可以一直写入文件


2. python中argparse如何传入bool值

>解决办法：
>
>```
># 在python文件中输入下面命令
>parser.add_argument('--balance', action='store_true')
>```
>
>现在要有balance的时候进行数据平衡，否则不进行平衡，使用方法为如下， 只要在命令
>行参数中不写入这个参数，就会传入False
>
>```
>python3 test.py --balance  # 进行数据平衡
>python3 test.py            # 不进行数据平衡
>```
>

3. 如何后台挂起，同时等待后台任务完成后再执行后面的任务

> 解决办法:
>
>在程序后面加入`&` 就会挂起，使用`wait $!` 等待上一个进程
