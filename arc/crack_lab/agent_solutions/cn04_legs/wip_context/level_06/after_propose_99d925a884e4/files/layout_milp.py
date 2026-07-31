"""MILP for legal landmark-cover layouts on the verified level-6 pieces."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from layout_search_clean import ORIENTATIONS, shift


LOW = -5
HIGH = 5


placements = []
by_piece = {index: [] for index in range(1, 5)}
for index in range(1, 5):
    for turns, (shape, green) in enumerate(ORIENTATIONS[index]):
        for row in range(LOW, HIGH + 1):
            for col in range(LOW, HIGH + 1):
                item = {
                    "piece": index,
                    "turns": turns,
                    "row": row,
                    "col": col,
                    "shape": shift(shape, row, col),
                    "green": shift(green, row, col),
                }
                item["body"] = item["shape"] - item["green"]
                by_piece[index].append(len(placements))
                placements.append(item)


anchor_shape, anchor_green = ORIENTATIONS[0][0]
anchor_body = anchor_shape - anchor_green
rows = []
cols = []
data = []
lower = []
upper = []


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
    coefficients = {
        variable: 1
        for variable, placement in enumerate(placements)
        if point in placement["body"]
    }
    add(coefficients, -np.inf, 0 if point in anchor_body else 1)

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
        add(coefficients, -np.inf, 1 if point in anchor_shape else 0)

matrix = coo_matrix(
    (data, (rows, cols)), shape=(len(lower), len(placements))
).tocsr()
result = milp(
    c=np.array([
        abs(item["row"]) + abs(item["col"]) for item in placements
    ], dtype=float),
    integrality=np.ones(len(placements)),
    bounds=Bounds(0, 1),
    constraints=LinearConstraint(matrix, lower, upper),
    options={"time_limit": 30},
)
print("status", result.status, result.message)
if result.x is not None:
    answer = {0: (0, 0, 0)}
    for variable, value in enumerate(result.x):
        if value > 0.5:
            item = placements[variable]
            answer[item["piece"]] = (
                item["turns"], item["row"], item["col"]
            )
    print("layout", answer)
    occupied = set(anchor_shape)
    for index in range(1, 5):
        turns, row, col = answer[index]
        occupied.update(shift(ORIENTATIONS[index][turns][0], row, col))
    print("bounds", (
        min(row for row, _ in occupied),
        min(col for _, col in occupied),
        max(row for row, _ in occupied),
        max(col for _, col in occupied),
    ), "area", len(occupied))
