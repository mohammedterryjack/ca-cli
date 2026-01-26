"""Multilayered Feedforward Neural Network using Pseudoinverse"""

from pathlib import Path

from numpy import ndarray, argmax, where
from numpy.random import seed, uniform
from numpy.linalg import pinv
from numpy import savez, load

from tools.utils.learn_utils.activation import (
    Activation,
    inverse_activation,
    activation_function,
)


class FFNN_ONESHOT:
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int],
        output_dimension: int,
        activation: Activation,
        **kwargs,
    ) -> None:
        seed(42)
        self.d_o = output_dimension
        self.dims = [input_dimension] + hidden_dimensions + [output_dimension]
        self.activation = activation_function(activation)
        self.inverse_activation = inverse_activation(activation)
        R = uniform(low=-0.1, high=0.1, size=(max(self.dims[:-1]), max(self.dims[1:])))
        self.Ws = list(map(lambda d1, d2: R[:d1, :d2], self.dims[:-1], self.dims[1:]))

    def fit(self, X: ndarray, Y: ndarray) -> None:
        for i in range(len(self.Ws)):
            Y_hat_next = (
                self.backward(
                    Y=Y, Ws=self.Ws[i + 1 :], inverse_activation=self.inverse_activation
                )
                if i < len(self.Ws)
                else Y
            )
            Y_hat = self.forward(X=X, Ws=self.Ws[:i], activation=self.activation)
            self.Ws[i] = pinv(Y_hat) @ Y_hat_next

    def predict(self, X: ndarray) -> ndarray:
        return self.forward(X=X, Ws=self.Ws, activation=self.activation)

    @staticmethod
    def backward(
        Y: ndarray, Ws: list[ndarray], inverse_activation: callable
    ) -> ndarray:
        Y_hat = Y
        for W in Ws[::-1]:
            Y_hat = inverse_activation(Y_hat) @ pinv(W)
        return Y_hat

    @staticmethod
    def forward(
        X: ndarray,
        Ws: list[ndarray],
        activation: callable,
    ) -> ndarray:
        Y_hat = X
        for W in Ws:
            Y_hat = activation(Y_hat @ W)
        return Y_hat

    def save(self, path: Path) -> None:
        savez(str(path), *self.Ws)

    def load(self, path: Path) -> None:
        data = load(str(path))
        self.Ws = [data[f"arr_{i}"] for i in range(len(data.files))]
