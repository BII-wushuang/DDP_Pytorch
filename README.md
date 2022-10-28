# Distributed training with pytorch

This code is suitable for multi-gpu training on a single machine. You can simply modify the GPUs that you wish to use in [train.sh](./train.sh). To commence training, execute
```
. train.sh
```

If your original execution script contains arguments, e.g. of the form `python train.py args_1 args_2`, simply transfer the arguments. For example,
```
. train.sh --epoch=10 --lr=0.002
```
