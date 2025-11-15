"""
Multilayered Feedforward Neural Network using Gradient Descent (via sklearn)
"""

from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor

from utils.learn_utils.activation import Activation, activation_function


class FFNN_ITERATIVE:
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int],
        output_dimension: int,
        activation: Activation,
    ) -> None:
        # Map Activation enum to sklearn activation string
        activation_map = {
            Activation.RELU: "relu",
            Activation.TAN: "tanh",
        }
        activation_str = activation_map.get(activation, "relu")

        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_dimensions),
            activation=activation_str,
            solver="adam",        # uses gradient descent with adaptive learning
            max_iter=5000,
            learning_rate_init=0.001,
            random_state=42,
            verbose=True,
        )
        self.input_dim = input_dimension
        self.output_dim = output_dimension
        self.activation = activation

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the MLP using gradient descent."""
        # sklearn expects 2D arrays; ensure shapes are right
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self.model.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass (predict output)."""
        X = np.asarray(X, dtype=float)
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Save the model weights and architecture."""
        from joblib import dump
        dump(self.model, path)
    
    def load(self, path: Path) -> None:
        """Load model weights and architecture."""
        from joblib import load
        self.model = load(path)

