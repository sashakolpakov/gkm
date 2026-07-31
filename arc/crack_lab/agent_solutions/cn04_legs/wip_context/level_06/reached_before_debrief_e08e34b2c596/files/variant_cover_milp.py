"""Find actual-action layouts with covered landmarks and no body clashes."""
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from variant_layout_clean import SHAPES
from variant_layout_search import GREENS, moved


ANCHOR_TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LOW, HIGH = -5, 5

placements = []
by_piece = {index: [] for index in range(1, 5)}
for index in range(1, 5):
    for turns, (shape, green) in enumerate(zip(SHAPES[index], GREENS[index])):
        for row in range(LOW, HIGH + 1):
            for col in range(LOW, HIGH + 1):
                item = {
                    "piece": index,
                    "turns": turns,
                    "row": row,
                    "col": col,
                    "shape": moved(shape, row, col),
                    "green": moved(green, row, col),
                }
                item["body"] = item["shape"] - item["green"]
                by_piece[index].append(len(placements))
                placements.append(item)

anchor_shape = SHAPES[0][ANCHOR_TURNS]
anchor_green = GREENS[0][ANCHOR_TURNS]
anchor_body = anchor_shape - anchor_green
rows, cols, data, lower, upper = [], [], [], [], []


def add(coefficients, lo, hi):
    row = len(lower)
    for col, value in coefficients.items():
        rows.append(row)
        cols.append(col)
        data.append(value)
    lower.append(lo)
    upper.append(hi)


for index in range(1, 5):
    add({variable: 1 for variable in by_piece[index]}, 1, 1)

all_points = set(anchor_body)
for placement in placements:
    all_points.update(placement["body"])
for point in all_points:
    add(
        {
            variable: 1
            for variable, placement in enumerate(placements)
            if point in placement["body"]
        },
        -np.inf,
        0 if point in anchor_body else 1,
    )

coverers = {}
for variable, placement in enumerate(placements):
    for point in placement["shape"]:
        coverers.setdefault(point, []).append(variable)

for point in anchor_green:
    add({variable: 1 for variable in coverers.get(point, [])}, 1, np.inf)

for variable, placement in enumerate(placements):
    for point in placement["green"]:
        coefficients = {variable: 1}
        for other in coverers.get(point, []):
            if placements[other]["piece"] != placement["piece"]:
                coefficients[other] = coefficients.get(other, 0) - 1
        add(
            coefficients,
            -np.inf,
            1 if point in anchor_shape else 0,
        )

matrix = coo_matrix(
    (data, (rows, cols)), shape=(len(lower), len(placements))
).tocsr()
result = milp(
    c=np.array(
        [
            abs(item["row"]) + abs(item["col"])
            for item in placements
        ],
        dtype=float,
    ),
    integrality=np.ones(len(placements)),
    bounds=Bounds(0, 1),
    constraints=LinearConstraint(matrix, lower, upper),
    options={"time_limit": 30},
)
print("status", result.status, result.message)
if result.x is not None:
    answer = {0: (ANCHOR_TURNS, 0, 0)}
    for variable, value in enumerate(result.x):
        if value > 0.5:
            item = placements[variable]
            answer[item["piece"]] = (
                item["turns"],
                item["row"],
                item["col"],
            )
    occupied = set(anchor_shape)
    for index in range(1, 5):
        turns, row, col = answer[index]
        occupied.update(moved(SHAPES[index][turns], row, col))
    print("layout", answer)
    print(
        "bounds",
        (
            min(row for row, _ in occupied),
            min(col for _, col in occupied),
            max(row for row, _ in occupied),
            max(col for _, col in occupied),
        ),
        "area",
        len(occupied),
    )
