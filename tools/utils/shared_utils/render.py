import os
from pathlib import Path
from numpy import array


def line_plot(data: list[float], width: int = 60) -> str:
    if len(data) == 0:
        return ""

    ticks = "▁▂▃▄▅▆▇█"

    if len(data) > width:
        step = len(data) / width
        data = [data[int(i * step)] for i in range(width)]

    mn, mx = min(data), max(data)
    rng = mx - mn if mx > mn else 1e-9

    spark = "".join(ticks[int((x - mn) / rng * (len(ticks) - 1))] for x in data)
    return spark


def bar_plot(scores: list[float], width: int = 40):
    """
    Plot similarity values vertically as ASCII bars in the terminal.
    Each score should be in [0, 1].
    """
    for i, s in enumerate(scores):
        # Clamp scores to [0, 1]
        s = max(0.0, min(1.0, s))
        bar_len = round(s * width)
        bar = "█" * bar_len
        print(f"{i:3d} | {bar:<{width}}  {s:.2f}")


def combine_side_by_side(left: str, right: str) -> str:
    left_lines = left.splitlines()
    right_lines = right.splitlines()

    max_len = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_len - len(left_lines))
    right_lines += [""] * (max_len - len(right_lines))
    combined = ""
    for l, r in zip(left_lines, right_lines):
        combined += f"{l:<40} | {r}\n"
    return combined


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
