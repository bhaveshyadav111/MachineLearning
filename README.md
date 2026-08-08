# Perceptron Trick

A from-scratch NumPy implementation of the **Perceptron Learning Algorithm** (the "trick" — weight update via misclassified point, not gradient descent on a loss function) on a synthetic 2D binary classification dataset.

## What's in the notebook

1. **Data generation** — `make_classification` from sklearn creates 600 samples, 2 features, 1 informative feature, 2 linearly separable clusters.
2. **Visualization** — scatter plot of the two classes colored by label.
3. **Perceptron trick** — the core algorithm (see below).
4. **Decision boundary extraction** — convert learned weights into a `y = mx + b` line and plot it over the data.

## The algorithm

```python
def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)   # bias trick: prepend column of 1s
    w = np.ones(X.shape[1])          # init weights (including bias) to 1
    lr = 0.1

    for epoch in range(1000):
        nth = np.random.randint(X.shape[0])   # pick one random point

        net = np.dot(X[nth], w)
        ypred = sign(net)
        w = w + lr * (y[nth] - ypred) * X[nth]  # update only on this point

    return w[0], w[1:]   # intercept, coefficients

def sign(net):
    return np.where(net > 0, 1, 0)
```

### Why this is called a "trick" and not gradient descent

There's no explicit loss function being differentiated here. The update rule

```
w ← w + lr * (y - ŷ) * x
```

is applied **per misclassified sample**, not as a batch or averaged gradient. When a point is correctly classified, `(y - ŷ) = 0` and `w` doesn't change. When it's misclassified:

- If true label is 1 but predicted 0 → `(y - ŷ) = 1` → weight vector nudged **toward** that point (rotates decision boundary to include it on the positive side).
- If true label is 0 but predicted 1 → `(y - ŷ) = -1` → weight vector nudged **away** from that point.

This is the original 1958 Rosenblatt perceptron update rule. It's guaranteed to converge in finite steps **only if the data is linearly separable** (Perceptron Convergence Theorem). No convergence guarantee otherwise — it will oscillate forever, which is why this notebook uses a fixed epoch count (1000) rather than a convergence check.

### Bias trick

Instead of maintaining a separate bias term `b` and updating it separately, a constant `1` is inserted as an extra feature column (`X = np.insert(X, 0, 1, axis=1)`). This folds the bias into the weight vector as `w[0]`, so the same update rule handles both weights and bias uniformly.

### Sign / step activation

```python
sign(net) = 1 if net > 0 else 0
```

This is a hard threshold — not the classic ±1 perceptron sign function, but a 0/1 version compatible with labels from `make_classification` (which are `{0, 1}`).

## Decision boundary math

The perceptron learns `w0 + w1*x1 + w2*x2 = 0` (the separating hyperplane). Solving for `x2` in terms of `x1` gives the plottable line:

```
x2 = -(w1/w2) * x1 - (w0/w2)
```

which in the notebook is:

```python
m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])
```

Then `y_input = m * x_input + b` is plotted over `x_input = np.linspace(-3, 3, 100)`.

## Known limitations in this implementation

- **Fixed epoch count, no convergence check** — runs exactly 1000 iterations regardless of whether the data is already perfectly classified. No early stopping.
- **No shuffling/epoch structure** — samples are drawn via `np.random.randint`, so it's not a proper "epoch = one pass over all data" loop; some points may never be sampled, others sampled many times.
- **No train/test split** — decision boundary is fit and evaluated on the same full dataset. Fine for a learning demo, not for evaluating generalization.
- **Weight initialization to `np.ones`** — arbitrary; doesn't affect convergence on separable data but is worth noting versus zero-init or random-init conventions.
- **No accuracy/metric reporting** — the notebook doesn't compute training accuracy or misclassification count after training.
