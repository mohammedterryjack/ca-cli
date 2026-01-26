import math
from numpy import array, array_equal


def first_differing_row(values1: list[list[int]], values2: list[list[int]]) -> int:
    mat1 = array(values1)
    mat2 = array(values2)
    i = 0
    for i, (row1, row2) in enumerate(zip(mat1, mat2, strict=True)):
        if not array_equal(row1, row2):
            return i
    return i


def cosine_similarity_between_evolutions(
    a: list[list[int]], b: list[list[int]]
) -> list[float]:
    """
    Compute cosine similarity between corresponding rows of two 2D integer lists.
    Returns a list of floats (one per row).
    """
    if len(a) != len(b):
        raise ValueError("Arrays must have the same number of rows.")

    sims = []
    for row_a, row_b in zip(a, b):

        if len(row_a) != len(row_b):
            raise ValueError("Row lengths do not match.")

        # Compute dot product
        dot = sum(x * y for x, y in zip(row_a, row_b))

        # Compute magnitudes
        mag_a = math.sqrt(sum(x * x for x in row_a))
        mag_b = math.sqrt(sum(y * y for y in row_b))

        # Handle zero-vector cases
        if mag_a == 0 or mag_b == 0:
            sims.append(0.0)
        else:
            sims.append(dot / (mag_a * mag_b))

    return sims
