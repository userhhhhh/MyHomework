# Guide

## dataset下载

### mnist_m

```
pip install datasets
cp /home/featurize/work/DRANet/script/download_mnist_m.py /home/featurize/data
python /home/featurize/data/download_mnist_m.py
```
然后将 data 文件夹移动到 DRANet 文件夹目录下

### cityscape

```
cd data/cityscape
git clone https://github.com/mcordts/cityscapesScripts
加载
```

## 改进的采用

见 `idea.md`

## 环境配置

pip install tensorboardX scipy prettytable

## Train

输入：task(clf or seg), datasets(M, MM, U, G, C), and experiment name.
```
python train.py -T [task] -D [datasets] --ex [experiment_name]
example) python train.py -T clf -D M MM --ex M2MM
```

## Test

输入：实验名称 + test 哪一步的 ckpt
```
python test.py -T [task] -D [datasets] --ex [experiment_name (that you trained)] --load_step [specific iteration]
example) python test.py -T clf -D M MM --ex M2MM --load_step 10000
```

## Tensorboard

看到所有训练结果，这个只依赖于tensorboard文件夹
```
cd to DRANet
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir tensorboard --bind_all
```
