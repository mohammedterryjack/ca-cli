from pydantic import BaseModel
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingStats(BaseModel):
    iteration: int
    exactness: float
    sparsity: float
    total: float
    quantisation_threshold: float


def plot_mask_loss(losses: list[TrainingStats], path: Path) -> None:
    iterations = [stat.iteration for stat in losses]
    exactness = [stat.exactness for stat in losses]
    sparsity = [stat.sparsity for stat in losses]
    total = [stat.total for stat in losses]

    fig = plt.figure(figsize=(8, 5)) 
    ax = fig.gca() 
    ax.plot(iterations, exactness, label="Exactness (MSE)") # Using ax.plot is safer
    ax.plot(iterations, sparsity, label="Sparsity (L1)")
    ax.plot(iterations, total, label="Total Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
