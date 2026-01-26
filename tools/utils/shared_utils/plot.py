

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_as_recurrence_map(timeseries:list[float], save_path:Path|None=None) -> None:
    ts = np.asarray(timeseries)

    if ts.ndim != 1:
        raise ValueError("Input timeseries must be 1D")
    if len(ts) < 2:
        raise ValueError("Timeseries must contain at least 2 points")

    x_t = ts[:-1]
    x_t1 = ts[1:]

    fig, ax = plt.subplots()

    ax.scatter(x_t, x_t1, s=10, alpha=0.7)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t+1)")
    ax.grid(False)

    if save_path:
        fig.savefig(str(save_path),  bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


