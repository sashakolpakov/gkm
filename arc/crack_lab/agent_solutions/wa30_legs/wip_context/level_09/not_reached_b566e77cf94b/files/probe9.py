"""Compact symbolic observations using only the documented public surface."""

import gkm_try

from perception import arr, bounded_bfs, connected_components


def glyph(cell):
    colors = set(int(value) for value in cell.flat)
    if 14 in colors:
        return "A"
    if 15 in colors:
        return "B"
    if 12 in colors:
        return "C"
    if 4 in colors:
        return "o"
    if colors == {5}:
        return "#"
    if 5 in colors:
        return "+"
    if 2 in colors and 9 in colors:
        return "T"
    if 2 in colors:
        return "="
    if 7 in colors:
        return "|"
    return "."


def tile_map(frame):
    grid = arr(frame)
    return [
        "".join(glyph(grid[row:row + 4, col:col + 4])
                for col in range(0, 64, 4))
        for row in range(0, 64, 4)
    ]


def actors(frame):
    return {
        color: [blob.bbox for blob in connected_components(
            frame, colors=(color,), min_area=4)]
        for color in (4, 12, 14, 15)
    }


def cell_positions(frame, color):
    positions = []
    for blob in connected_components(frame, colors=(color,), min_area=4):
        positions.append((round(blob.centroid[0] / 4, 1),
                          round(blob.centroid[1] / 4, 1), blob.area))
    return positions


def metric(frame):
    grid = arr(frame)
    empty_targets = []
    cargo_cells = []
    for row in range(16):
        for col in range(16):
            colors = set(int(value) for value in
                         grid[row * 4:row * 4 + 4,
                              col * 4:col * 4 + 4].flat)
            if 2 in colors and 9 in colors:
                empty_targets.append((row, col))
            if 4 in colors and 9 in colors:
                cargo_cells.append((row, col))
    return {"empty": empty_targets, "cargo": cargo_cells}


def outcome(env, actions):
    clone = env.clone()
    before = clone.levels_completed
    for action in actions:
        if clone.terminal():
            break
        clone.step(action)
    return {
        "path": "".join(str(action) for action in actions),
        "reward": clone.levels_completed - before,
        "metric": metric(clone.frame()),
        "actors": actors(clone.frame()),
    }


def inspect(env):
    gkm_try.resumed_solve(env)
    print("STATE", {
        "levels": env.levels_completed,
        "terminal": env.terminal(),
        "actions": env.actions,
    })
    print("MAP")
    print("\n".join(tile_map(env.frame())))
    print("ACTORS", actors(env.frame()))
    for path in (
        [1], [2], [3], [4], [5],
        [1] * 4, [2] * 4, [3] * 4, [4] * 4, [5] * 4,
        [1, 3, 5], [3, 3, 1, 1, 5],
    ):
        print("PROBE", outcome(env, path))
    idle = env.clone()
    for turn in range(121):
        if idle.terminal():
            break
        if turn % 8 == 0:
            print("IDLE", turn, {
                "reward": idle.levels_completed - env.levels_completed,
                "metric": metric(idle.frame()),
                "couriers12": cell_positions(idle.frame(), 12),
                "courier15": cell_positions(idle.frame(), 15),
            })
        idle.step(5)
    for path in (
        [1, 3, 5, 1, 1, 5],
        [1, 3, 5, 1, 1, 5,
         2, 2, 3, 3, 3, 1, 5, 4, 1, 5],
        [1, 3, 5, 1, 1, 5,
         2, 2, 3, 3, 3, 1, 5, 4, 1, 5] + [5] * 44,
        [1, 3, 5] + [3] * 8 + [1] * 3 + [5],
        [1, 3, 5] + [3] * 8 + [1] * 3 + [5, 2],
        ([1, 3, 5] + [3] * 8 + [1] * 3 + [5, 2]
         + [2, 2] + [4] * 5 + [1, 5, 2] + [3] * 5
         + [1] * 3 + [5, 2]),
        ([1, 3, 5] + [3] * 8 + [1] * 3 + [5, 2]
         + [2, 2] + [4] * 5 + [1, 5, 2] + [3] * 5
         + [1] * 3 + [5, 2]
         + [2] + [4] * 4 + [1] * 3 + [5] + [2] * 3
         + [3] * 4 + [1] * 4 + [5, 2]),
        ([1, 3, 5] + [3] * 6 + [5, 4]
         + [4] * 2 + [1, 5, 2] + [3] * 4 + [5, 4]),
        ([1, 3, 5] + [3] * 6 + [5, 4]
         + [4] * 2 + [1, 5, 2] + [3] * 4 + [5, 4]
         + [5] * 38),
        [3] * 11 + [5] * 20,
    ):
        print("TARGET", outcome(env, path))
    for waits in range(10, 21):
        result = outcome(env, [3] * 11 + [5] * waits)
        print("DISMISS", waits, not result["actors"][15],
              result["metric"]["empty"])
    for waits in range(4, 13):
        result = outcome(env, [3] * 12 + [2] * 3 + [5] * waits)
        print("MAZE", waits, not result["actors"][15],
              result["actors"][15], result["actors"][14])
    first_delivery = [1, 3, 5] + [3] * 8 + [1] * 3 + [5]
    for waits in range(4, 10):
        result = outcome(
            env, first_delivery + [2] * 4 + [3] * 3 + [5] * waits)
        print("COMBO", waits, not result["actors"][15],
              result["metric"]["empty"], result["actors"][14])
    combo = first_delivery + [2] * 4 + [3] * 3 + [5] * 7
    print("COMBOSTATE", outcome(env, combo))
    finish = [3] * 3 + [1, 5] + [4] * 6 + [1] * 3 + [5, 2]
    for waits in (0, 5, 10, 15):
        result = outcome(env, combo + finish + [5] * waits)
        print("FINISH", waits, result["reward"], result["metric"]["empty"],
              result["actors"][15])
    two_deliveries = (
        [1, 3, 5] + [3] * 8 + [1] * 3 + [5, 2]
        + [2, 2] + [4] * 5 + [1, 5, 2] + [3] * 5
        + [1] * 3 + [5, 2]
    )
    for contact in ([5], [3, 5], [3, 5, 5]):
        result = outcome(env, two_deliveries + contact)
        print("CONTACT", contact, not result["actors"][15],
              result["actors"][15], result["metric"]["empty"])
    for chase in range(2, 9):
        result = outcome(env, two_deliveries + [3] * chase + [5])
        print("CHASE", chase, not result["actors"][15],
              result["actors"][15], result["actors"][14])
    for contact in ([3] * 5 + [2, 5],
                    [3] * 5 + [2, 5, 5],
                    [3] * 5 + [2, 2, 5],
                    [3] * 4 + [2, 5]):
        result = outcome(env, two_deliveries + contact)
        print("CORNER", contact, not result["actors"][15],
              result["actors"][15], result["actors"][14])
    take_lower_left = (
        [3] * 3 + [1, 5] + [4] * 5 + [1] * 3 + [5, 2]
        + [3] * 5 + [5] + [4] * 6 + [1] * 3 + [5]
    )
    result = outcome(env, combo + take_lower_left)
    print("TWOLEFT", result["reward"], result["metric"]["empty"],
          result["actors"])
    take_right = (
        [4] * 8 + [1, 1, 5, 2] + [3] * 5 + [1] * 3 + [5]
    )
    for waits in (0, 5, 10):
        result = outcome(env, combo + take_right + [5] * waits)
        print("RIGHTFIN", waits, result["reward"],
              result["metric"]["empty"], result["actors"][15])
    stage_then_dismiss = (
        [1, 3, 5] + [3] * 6 + [5, 4]
        + [2] + [3] * 6 + [5] * 11
    )
    for waits in (0, 5, 10):
        result = outcome(
            env, stage_then_dismiss + take_right + [5] * waits)
        print("STAGEFIN", waits, result["reward"],
              result["metric"]["empty"], result["actors"][15])
    alt_combo = (
        [1, 3, 5] + [3] * 7 + [1] * 3 + [5]
        + [2] * 4 + [3] * 4 + [5] * 7
    )
    for waits in (0, 5, 10):
        result = outcome(env, alt_combo + take_right + [5] * waits)
        print("ALTFIN", waits, result["reward"],
              result["metric"]["empty"], result["actors"][15])
    visit_static = [1] * 8 + [3] * 4 + [5]
    for waits in (0, 2, 5, 10):
        result = outcome(env, visit_static + [5] * waits)
        print("STATIC", waits, result["actors"][12],
              result["metric"]["empty"])
    prestage = [1, 3, 5] + [3] * 11 + [5, 2]
    for contact in ([5] * 13, [5] * 14, [5] * 15,
                    [5] * 11 + [3, 5],
                    [5] * 10 + [3, 5, 5]):
        result = outcome(env, prestage + contact)
        print("PRESTAGE", len(contact), not result["actors"][15],
              result["actors"][15], result["actors"][14],
              result["metric"]["empty"], result["actors"][4])
    prestage_four = [1, 3, 5] + [3] * 10 + [5, 2]
    for contact in ([5] * 12 + [3, 5],
                    [5] * 11 + [3, 3, 5],
                    [5] * 13 + [3, 5]):
        result = outcome(env, prestage_four + contact)
        print("PREFOUR", len(contact), not result["actors"][15],
              result["actors"][15], result["actors"][14],
              result["metric"]["empty"], result["actors"][4])
    prefour_success = prestage_four + [5] * 12 + [3, 5]
    recover_staged = [1, 5] + [4] * 2 + [1] * 3 + [5]
    fetch_second_right = (
        [2] * 2 + [4] * 6 + [1, 5, 2]
        + [3] * 5 + [1] * 3 + [5]
    )
    for waits in (0, 3):
        result = outcome(
            env, prefour_success + recover_staged
            + fetch_second_right + [5] * waits)
        print("COMPACT", waits, result["reward"],
              result["metric"]["empty"], result["actors"][15])
    relocate_center = [4] + [1] * 4 + [5] + [4] * 2 + [5]
    fetch_after_center = (
        [2] * 2 + [4] * 5 + [1, 5, 2]
        + [3] * 5 + [1] * 4 + [5]
    )
    print("CENTERSTEP", outcome(env, combo + relocate_center))
    for waits in (0, 2):
        result = outcome(
            env, combo + relocate_center + fetch_after_center + [5] * waits)
        print("CENTERFIN", waits, result["reward"],
              result["metric"]["empty"], result["actors"][15])
    for waits in (0, 5, 9):
        result = outcome(
            env, prefour_success + take_right + [5] * waits)
        print("LEAVESTAGE", waits, result["reward"],
              result["metric"]["empty"], result["actors"][4])
    stage_left_for_courier = (
        [3] * 3 + [1, 5] + [4] * 3 + [1, 5]
    )
    fetch_from_staging_lane = (
        [4] * 8 + [1, 5, 2] + [3] * 5 + [1] * 3 + [5]
    )
    result = outcome(
        env, combo + stage_left_for_courier + fetch_from_staging_lane)
    print("COOPERATE", result["reward"], result["metric"]["empty"],
          result["actors"])
    for chase in range(2, 7):
        contact = [2] + [3] * chase + [5]
        result = outcome(env, two_deliveries + contact)
        print("CUTCHASE", chase, not result["actors"][15],
              result["actors"][15], result["actors"][14])
    cut_contact = [2] + [3] * 4 + [5]
    for rises in (3, 4):
        last_local = [3] * 2 + [5] + [4] * 6 + [1] * rises + [5]
        waits = 59 - len(two_deliveries + cut_contact + last_local)
        result = outcome(
            env, two_deliveries + cut_contact + last_local + [5] * waits)
        print("CUTFIN", rises, waits, result["reward"],
              result["metric"]["empty"], result["actors"])
    first_alt = [1, 3, 5] + [3] * 7 + [1] * 3 + [5]
    second_alt = (
        [2] * 3 + [4] * 4 + [1, 5, 2]
        + [3] * 6 + [1] * 3 + [5]
    )
    alt_two = first_alt + second_alt
    print("ALTTWO", outcome(env, alt_two))
    for contact in (
        [2, 3, 3, 3, 3, 5],
        [2, 2, 3, 3, 3, 5],
        [3, 3, 3, 2, 5],
        [2, 3, 3, 3, 5],
        [2, 3, 3, 5],
        [3, 2, 3, 5],
        [3, 3, 2, 5],
        [2, 2, 3, 5],
    ):
        result = outcome(env, alt_two + contact)
        print("ALTCUT", contact, not result["actors"][15],
              result["actors"][15], result["actors"][14],
              result["metric"]["empty"])
    alt_cut = [2, 3, 3, 3, 5]
    print("ALTCUTSTATE", outcome(env, alt_two + alt_cut))
    for contact in ([3, 1, 5], [3, 1, 5, 5],
                    [1, 3, 5], [3, 5]):
        result = outcome(env, alt_two + alt_cut + contact)
        print("HELPERCONTACT", contact, result["actors"][12],
              result["actors"][4], result["actors"][14])
    alt_last = [3] * 2 + [5] + [4] * 6 + [1] * 3 + [5]
    for waits in (0, 4, 7):
        result = outcome(env, alt_two + alt_cut + alt_last + [5] * waits)
        print("ALTWIN", waits, result["reward"],
              result["metric"]["empty"], result["actors"])
    take_lower_of_pair = (
        [2] + [3] * 2 + [5] + [4] * 6 + [1] * 3 + [5]
    )
    for waits in (0, 3, 6):
        result = outcome(
            env, alt_two + alt_cut + take_lower_of_pair + [5] * waits)
        print("LOWERWIN", waits, result["reward"],
              result["metric"]["empty"], result["actors"])
    stage_lower = [2] + [3] * 2 + [5] + [4] * 2 + [1, 5]
    fill_vacated_center = [3] * 3 + [5] + [4] * 4 + [1] * 2 + [5]
    result = outcome(
        env, alt_two + alt_cut + stage_lower
        + fill_vacated_center + [5])
    print("FINALCOOP", result["reward"], result["metric"]["empty"],
          result["actors"])
    print("STAGELOWERSTATE", outcome(env, alt_two + alt_cut + stage_lower))
    stage_below = [2, 3, 3, 5, 2, 5, 4]
    fill_from_below = (
        [1] + [3] * 2 + [1, 5] + [4] * 4 + [1] * 2 + [5]
    )
    for waits in (0, 1):
        result = outcome(
            env, alt_two + alt_cut + stage_below
            + fill_from_below + [5] * waits)
        print("BELOWCOOP", waits, result["reward"],
              result["metric"]["empty"], result["actors"])
    dismiss_state = env.clone()
    for action in alt_two:
        dismiss_state.step(action)
    shortest_dismissal = bounded_bfs(
        dismiss_state,
        lambda node, _: not connected_components(
            node.frame(), colors=(15,), min_area=4),
        max_states=4000,
        max_depth=5,
    )
    print("SHORTEST_DISMISSAL", shortest_dismissal)
    preserve_dismissal = bounded_bfs(
        dismiss_state,
        lambda node, _: (
            not connected_components(
                node.frame(), colors=(15,), min_area=4)
            and len([
                cell for cell in metric(node.frame())["empty"]
                if cell in {
                    (3, 5), (3, 6), (3, 7), (4, 5), (4, 7),
                    (5, 5), (5, 6), (5, 7),
                }
            ]) <= 2
        ),
        max_states=12000,
        max_depth=9,
    )
    print("PRESERVE_DISMISSAL", preserve_dismissal)



gkm_try.A.run_program("wa30", inspect)
