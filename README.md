# ML_Homework

## Environment

`./DRANet/markdown/Guide.md`

## Improvements

`./DRANet/markdown/idea.md`

## Train

Input：task(clf or seg), datasets(M, MM, U, G, C), and experiment name.

```bash
python train.py -T [task] -D [datasets] --ex [experiment_name]
example) python train.py -T clf -D M MM --ex M2MM
```

## Test

Input：exp_name + ckpt

```bash
python test.py -T [task] -D [datasets] --ex [experiment_name (that you trained)] --load_step [specific iteration]
example) python test.py -T clf -D M MM --ex M2MM --load_step 10000
```

## Tensorboard

only rely upon tensorboard

```bash
cd to DRANet
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir tensorboard1 --bind_all
```
