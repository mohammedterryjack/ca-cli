from hashlib import sha256
from numpy import ndarray, array
from pathlib import Path
from PIL import Image
import numpy as np
from numpy import array
from pyarrow import Table
from pyarrow.parquet import write_table, read_table
from statistics import mode

COLUMN_NAME = "text"
BOS = "‖"
EOL = "|"
WILDCARD = "*"
SPECIAL = {
    "O": 0,
    "I": 1,
    "Z": 2,
    "E": 3,
    "A": 4,
    "S": 5,
    "b": 6,
    "T": 7,
    "B": 8,
    "g": 9,
}


def parquet_string_to_evolution_and_emergence(
    parquet_string: str,
) -> tuple[list[list[int]], list[list[bool]]]:
    lines = parquet_string.strip(BOS).split(EOL)

    evolution = []
    mask = []

    for line in lines:
        evo_row = []
        mask_row = []
        for ch in line:
            if ch.isdigit():
                evo_row.append(int(ch))
                mask_row.append(False)
            else:
                evo_row.append(SPECIAL[ch])
                mask_row.append(True)

        evolution.append(evo_row)
        mask.append(mask_row)
    return evolution, mask


def evolution_and_emergence_to_parquet_string(
    evolution: ndarray, emergence_mask: ndarray
) -> str:
    special = array(list(SPECIAL))
    base = evolution.astype(str)
    rendered = np.where(emergence_mask, special[evolution], base)
    return BOS + EOL.join("".join(row) for row in rendered) + BOS


def save_evolution(values: list[list[int]], path: Path) -> None:
    arr = array(values, dtype="uint8")
    N = arr.max()
    if N == 0:
        img_data = arr.astype("uint8")  # all zeros, no scaling needed
    else:
        img_data = (arr / N * 255).astype("uint8")
    img = Image.fromarray(img_data, mode="L")
    img.save(str(path))


def evolution_to_parquet_string(evolution: ndarray) -> str:
    return BOS + EOL.join("".join(map(str, row)) for row in evolution) + BOS


def parquet_string_to_evolution(parquet_string: str) -> list[list[int]]:
    return [[int(ch) for ch in row] for row in parquet_string.strip(BOS).split(EOL)]


def safe_string_to_evolution(
    generated_string: str, row_length: int | None = None, decode_special: bool = False
) -> list[list[int]]:
    generated_string = (
        generated_string.strip(BOS).replace(BOS, EOL).replace(WILDCARD, "")
    )
    # print(generated_string.replace(EOL,'\n'))
    if decode_special:
        evolution, _ = parquet_string_to_evolution_and_emergence(
            parquet_string=generated_string
        )
    else:
        evolution = parquet_string_to_evolution(parquet_string=generated_string)
    if row_length is None:
        row_lengths = [len(row) for row in evolution]
        normal_len = mode(row_lengths)
    else:
        normal_len = row_length

    normalized = []
    for i, row in enumerate(evolution):
        row_len = len(row)
        if row_len == normal_len:
            normalized.append(row)
            continue
        if row_len > normal_len:
            normalized.append(row[:normal_len])
            continue
        if row_len < normal_len:
            break
        break
    return normalized


def save_evolutions_as_parquet(
    evolutions: dict[str, ndarray],
    output_dir: Path,
    additional_features: dict[str, ndarray] | None = None,
) -> None:
    names = list(evolutions)
    if additional_features is None:
        evolution_strings = list(map(evolution_to_parquet_string, evolutions.values()))
    else:
        evolution_strings = [
            evolution_and_emergence_to_parquet_string(
                evolution=evolutions[name], emergence_mask=additional_features[name]
            )
            for name in names
        ]
    table = Table.from_arrays([names, evolution_strings], names=["name", COLUMN_NAME])
    parquet_path = output_dir.with_suffix(".parquet")
    write_table(table, parquet_path)


def load_evolutions_from_parquet(parquet_path: Path) -> dict[str, ndarray]:
    table = read_table(parquet_path)
    names = table.column("name").to_pylist()
    evolution_strings = table.column(COLUMN_NAME).to_pylist()
    return {
        name: array(parquet_string_to_evolution(parquet_string=evolution_str))
        for name, evolution_str in zip(names, evolution_strings, strict=True)
    }
