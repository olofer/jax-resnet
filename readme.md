# jax-resnet

Basic ResNet with built-in Layer Normalization, implemented in `jax`. Demonstration program executions are listed below.

## Examples

### Binary classification

```
python3 resnet-bce-example.py --show-loss --show-function --epochs 200 --layers 8 --weight-decay 1.0 --step-size 3e-4 --shuffle
```

### Density estimation via likelihood ratio trick

```
python3 resnet-lrt-example.py --show-loss --show-function --epochs 501 --layers 10 --step-size 4e-4 --shuffle
```

More layers appear to help here. Trickier to train the density-like function.

### Multivariate (least-squares) regression

```
python3 resnet-reg-example.py --show-loss --show-function --step-size 4e-3 --epochs 800 --layers 8
```

### Multi-class classification 

```
python3 resnet-mce-example.py  --show-loss --eval-function
```

Standard multi-class function is converted to rate-functions in this example. Special type of data-generating process where this makes sense (see code).

### Denoising Density Estimation (DDE)

```
python3 resnet-dde-example.py
```

(only illustrated with a plain normal distribution, so far; essentially Tweedie's formula).
