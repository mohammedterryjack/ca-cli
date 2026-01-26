from rich.text import Text
from tools.utils.shared_utils.render import colour

from rich.text import Text


def render_halfblock_rich(values: list[list[int]]) -> Text:
    t = Text()
    h, w = len(values), len(values[0])
    for y in range(0, h, 2):
        line = Text()
        for x in range(w):
            top_value = values[y][x]
            bottom_value = values[y + 1][x] if y + 1 < h else top_value
            r1, g1, b1 = colour(top_value + 1)
            r2, g2, b2 = colour(bottom_value + 1)
            line.append("▀", style=f"rgb({r1},{g1},{b1}) on rgb({r2},{g2},{b2})")
        t.append(line)
        t.append("\n")
    t.append("\n")
    return t
