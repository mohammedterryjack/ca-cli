from enum import Enum

from numpy import maximum, nan_to_num, tan, arctan


class Activation(Enum):
    IDENTITY = "identity"
    RELU = "rectified linear unit"
    TAN = "tan"


def activation_function(activation: Activation) -> callable:
    return {
        Activation.IDENTITY: lambda x: x,
        Activation.RELU: lambda x: maximum(0, x),
        Activation.TAN: tan,
    }[activation]


def inverse_activation(activation: Activation) -> callable:
    f = {
        Activation.TAN: arctan,
    }.get(activation, lambda x: x)
    return lambda x: nan_to_num(f(x), nan=0.0)
