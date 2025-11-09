import os
from PIL import Image
from pathlib import Path
from numpy import array


def save_evolution(values: list[list[int]], path: Path) -> None:
    arr = array(values, dtype="uint8")
    N = arr.max()
    if N == 0:
        img_data = arr.astype("uint8")  # all zeros, no scaling needed
    else:
    	img_data = (arr / N * 255).astype("uint8")
    img = Image.fromarray(img_data, mode="L")
    img.save(str(path))


def colour(i: int) -> tuple[int, int, int]:
    r = (i * 97) % 256
    g = (i * 151) % 256
    b = (i * 233) % 256
    return (r, g, b)


def render_halfblock(values: list[list[int]]) -> str:
    rows = []
    h = len(values)
    w = len(values[0])
    for y in range(0, h, 2):
        line = ""
        for x in range(w):
            top_value = values[y][x]
            if y + 1 < h:
                bottom_value = values[y + 1][x]
            else:
                bottom_value = top_value
            r1, g1, b1 = colour(top_value + 1)
            r2, g2, b2 = colour(bottom_value + 1)
            line += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀"
        line += "\033[0m"
        rows.append(line)
    return "\n".join(rows)
