import numpy as np

from finance_ai.network import MLP
from finance_ai.experiment import fit_scaler, metrics, split_positions, transform


def test_gradients_match_numerical_estimate():
    model = MLP(sizes=(3, 4, 1), seed=10)
    x = np.array([[0.4, -0.2, 0.7], [0.3, 0.1, -0.2]])
    y = np.array([1., 0.])
    _, dw, _ = model.gradients(x, y)
    original = model.weights[0][0, 0]
    eps = 1e-5
    model.weights[0][0, 0] = original + eps
    plus = model.gradients(x, y)[0]
    model.weights[0][0, 0] = original - eps
    minus = model.gradients(x, y)[0]
    model.weights[0][0, 0] = original
    assert np.isclose(dw[0][0, 0], (plus-minus)/(2*eps), atol=1e-6)


def test_training_reduces_loss():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(float)
    model = MLP(sizes=(3, 12, 1), seed=20)
    before = metrics(model, x, y)["loss"]
    for _ in range(100):
        model.train_batch(x, y, lr=0.01)
    assert metrics(model, x, y)["loss"] < before


def test_scaler_train_only():
    train = np.ones((10, 2, 3))
    other = np.full((4, 2, 3), 1000.)
    mu, sigma = fit_scaler(train)
    assert np.all(mu == 1) and np.all(sigma == 1)
    assert transform(other, mu, sigma).shape == (4, 6)
    assert split_positions(200) == (140, 170)
