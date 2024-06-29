# jax-resnet

Basic ResNet with built-in Layer Normalization, implemented in `jax`.

## Example binary classification

```
python3 resnet-bce-example.py --show-loss --show-function --epochs 200 --layers 8 --weight-decay 1.0 --step-size 3e-4 --shuffle
```

## Example density estimation via likelihood ratio trick

```
python3 resnet-lrt-example.py --show-loss --show-function --epochs 1001 --layers 12 --weight-decay 1.0 --step-size 2e-4 --shuffle
```

More layers appear to help here. Tricker to train the density-like function.

## Example multivariate regression

```
WIP
```
