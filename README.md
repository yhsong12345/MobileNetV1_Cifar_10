# MobileNetV1_Cifar_10

This is the practice for the implementation of MobileNetV1.  Models are trained with Cifar 10.

## Prerequisites
- Pytorch 2.0.1
- Python 3.11.4
- Window 11
- conda 23.7.4

## Training
```
# GPU training
python train.py -m Mobilenet0.5 -e 300 -lr 0.01 -b 128 -s 32 -d outputs
```

## Testing
```
python test.py -m Mobilenet0.5 -e 300 -lr 0.01 -b 128 -s 32 -d outputs
```

## Result (Accuracy)

Pretrained model should be downloaded if you click the name of Model.

| Model             | Acc.        | Param.        |
| ----------------- | ----------- |----------- |
| [Mobilenet]()          | 81.07%      |  3.22M     |
| [Mobilenet0.5]()          | 75.29%      |1.64M      |
| [Mobilenet0.25]()         | 75.3%      | 0.86M      |


## Plot
Plots are in the plots folder.
