from numpy import prod, dot, arccos, clip, ndarray, exp, linspace, pi, sin
from numpy.linalg import norm
from numpy.random import normal
 

def make_reference_vector(width:int, mode:str) -> ndarray:
    if mode == "random":
        u = random.normal(size=width)
    elif mode == "decay":
        u = exp(-arange(width)/20.0)
    elif mode == "sin":
        u = sin(linspace(0, 2*pi, width))
    return u / norm(u)

def embed_row_cosine(row:ndarray, u:ndarray) -> float:
    row = row.astype(float)
    dot_val = dot(row, u)
    denom = norm(row) * norm(u)
    cos_val = dot_val / denom
    cos_val = clip(cos_val, -1, 1)
    theta = arccos(cos_val)
    return float(theta / pi)

def evolution_to_1d_sequence_cosine(
    evolution: ndarray
) -> list[float]:
    _,W = evolution.shape 
    ref = make_reference_vector(width=W, mode="sin")
    trajectory =  [
        embed_row_cosine(row=row, u=ref) for row in evolution
    ]
    pmin = min(trajectory)
    pmax = max(trajectory)
    trajectory_scaled = [(p - pmin) / (pmax - pmin) for p in trajectory]
    return trajectory_scaled



def evolution_to_1d_sequence_gray(evolution: ndarray) -> list[float]:
    """
    Encode each row of a 2D integer array as a float in [0,1] using Gray code.
    
    evolution: 2D array of shape (T, W), T timesteps, W cells
    Returns: list of floats, length T
    """
    max_cell_state = int(evolution.max())
    total = (max_cell_state + 1) ** evolution.shape[1]  # number of possible states per row

    def row_to_gray(row):
        idx = 0
        factor = 1
        for a in row[::-1]:  # last cell first
            if (idx // factor) % 2 == 0:
                idx += a * factor
            else:
                idx += (max_cell_state - a) * factor
            factor *= (max_cell_state + 1)
        return idx / total

    return [row_to_gray(row) for row in evolution]
