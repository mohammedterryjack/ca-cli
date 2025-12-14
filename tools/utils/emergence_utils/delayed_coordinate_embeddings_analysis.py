from collections import Counter

from numpy import (
    array,
    asarray,
    arange,
    full,
    full_like,
    isnan,
    nan,
    nan_to_num,
    any as np_any,
)


def delay_embedding_density(
    sequence: list[int], time_delay: int, embedding_dimension: int, normalise: bool
) -> list[float]:
    sequence = asarray(sequence)
    N = len(sequence)
    start = (embedding_dimension - 1) * time_delay
    if N <= start:
        return full(N, nan)

    embedded = array(
        [
            tuple(sequence[i - arange(embedding_dimension) * time_delay])
            for i in range(start, N)
        ]
    )

    counts = Counter(map(tuple, embedded))
    densities = full(N, nan)
    for idx, vec in enumerate(embedded, start=start):
        densities[idx] = counts[tuple(vec)]

    if normalise:
        valid = ~isnan(densities)
        if np_any(valid):
            vals = densities[valid]
            densities[valid] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)

    return densities


def filter_spacetime_by_delay_embedding_density(
    spacetime: list[list[int]],
    time_delay: int,
    embedding_dimension: int,
    normalise: bool = True,
) -> list[list[float]]:
    data = asarray(spacetime)
    n_rows, n_cols = data.shape
    rareness_matrix = full_like(data, nan, dtype=float)

    for i in range(n_rows):
        sequence = data[i, :]
        rareness_matrix[i, :] = delay_embedding_density(
            sequence=sequence,
            time_delay=time_delay,
            embedding_dimension=embedding_dimension,
            normalise=normalise,
        )

    return rareness_matrix
