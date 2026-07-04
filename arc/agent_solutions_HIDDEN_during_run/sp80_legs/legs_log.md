# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1 (solved, 4 moves: 4,4,4,5)
- Game: liquid-pouring. Spout (color 4) + liquid (6) at top; ACTION5 pours the
  liquid straight down the spout column. Bar (color 9) is the avatar:
  ACTION1/2/3/4 = up/down/left/right, 4 cells per step; vertical range rows
  12..47. Cups (color 11) at the bottom with 4-wide rim openings.
- Mechanic: liquid landing on the bar spreads and spills off BOTH ends into the
  4-wide columns just outside each bar edge. Align bar so both spill columns
  equal the two cup openings, then pour once -> level complete.
- Hazards: a pour that misses is a strike (~5 strikes = GAME_OVER); top-row
  timer (color 14) depletes 2 cells per step (~32 step budget).
- Legs added: bbox, cup_openings, move_bar_to_left_col, best_deflect_left_col,
  pour. Player = best_deflect_left_col -> move_bar_to_left_col -> pour.
- Note for level 2: layout appears vertically flipped (spout at bottom, timer
  at row 63) -- gravity/pour direction likely inverted; legs may need an axis
  parameter.

## Recurring composition pattern (candidate higher-order leg)
- Only one player exists so far (play_level_1), so there is nothing to
  deduplicate yet -- the skills already live once each in legs.py and the
  player is already thin composition. Recording the SHAPE for when level 2+
  players arrive.
- Pattern: **sense -> align -> act**.
    target = <sense>(frame)      # read frame, compute a goal coordinate
    <align>(env, target)         # step the avatar until it reaches target
    <act>(env)                   # fire the terminal action once aligned
  In play_level_1 this is:
    best_deflect_left_col -> move_bar_to_left_col -> pour.
- Candidate HOL when a second player repeats it, e.g.
    def align_and_act(env, sense, align, act):
        align(env, sense(env.frame())); act(env)
  Players would then read: `align_and_act(env, best_deflect_left_col,
  move_bar_to_left_col, pour)`. Hold off until a 2nd instance confirms the
  shape (avoid a one-off abstraction).
