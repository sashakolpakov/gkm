"""
ARC-AGI-3 connector for the colimit-cone substrate (offline, stub-driven).

This module is the Engineer's scaffolding for pointing the colimit-cone
machinery at ARC-AGI-3. It is deliberately offline: no API key, no network. It
defines the data types matching the public API shape, a scene functor that
turns a frame into objects/relations/deltas, and the explicit correspondence

    ARC colour-slot   <->   cone-substrate channel
    azimuth to nearest object of a colour   <->   channel azimuth observation
    standing on such an object   <->   that channel reading HERE
    CALL(leg, colour)   <->   bind a channel-blind leg to a colour slot (v3)

so the SAME free-energy cone selection (cone_foraging_bound) applies without
change. A tiny deterministic stub game lets the whole loop run end to end with
reproducible fixtures, so the connector can be tested before any live API use.

API shape (arcprize ARC-AGI-3, as of 2026): a frame is a 64x64 grid of 4-bit
colour indices (0-15); actions are ACTION1..ACTION5 (simple) and ACTION6 with
(x, y) coordinates; every RESET/ACTION returns a snapshot with the latest
frame(s), cumulative score, and game state. See COLIMIT_CONE_APPROACH.md
Section 12.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import cone_foraging as cf
import cone_foraging_bound as cb

Cell = Tuple[int, int]  # (x, y)
Color = int

MAX_GRID = 64
NUM_COLORS = 16
BACKGROUND = 0  # ARC convention: colour 0 is the canvas background


class GameAction(IntEnum):
    """Mirror of the public action set. ACTION1-5 are simple; ACTION6 carries
    coordinates. We map the four simple directional actions onto grid moves and
    leave ACTION5 as a no-op/interact slot."""

    ACTION1 = 1  # UP
    ACTION2 = 2  # RIGHT
    ACTION3 = 3  # DOWN
    ACTION4 = 4  # LEFT
    ACTION5 = 5  # INTERACT / no-op
    ACTION6 = 6  # coordinate action (x, y)
    ACTION7 = 7  # UNDO (games that support it; see API docs)


# Correspondence between the cone substrate's moves and ARC simple actions.
MOVE_TO_ACTION = {
    cf.MOVE_NAMES.index("UP"): GameAction.ACTION1,
    cf.MOVE_NAMES.index("RIGHT"): GameAction.ACTION2,
    cf.MOVE_NAMES.index("DOWN"): GameAction.ACTION3,
    cf.MOVE_NAMES.index("LEFT"): GameAction.ACTION4,
    cf.MOVE_NAMES.index("STAY"): GameAction.ACTION5,
}
ACTION_TO_DELTA = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (1, 0),
    GameAction.ACTION3: (0, 1),
    GameAction.ACTION4: (-1, 0),
    GameAction.ACTION5: (0, 0),
}


class GameState(IntEnum):
    """Game lifecycle. The live REST API and the local arcengine report states
    by NAME; we accept every spelling either layer uses and collapse them onto
    these four values so the rest of the code can compare by identity.

    Aliases (same value => same member): the live API / arcengine call the
    "running" state ``NOT_FINISHED`` and the "not yet reset" state
    ``NOT_PLAYED``; older docs use ``IN_PROGRESS`` / ``NOT_STARTED``. They map to
    the same members, so ``GameState["NOT_FINISHED"] is GameState.IN_PROGRESS``
    and existing ``!= IN_PROGRESS`` / ``in (WIN, GAME_OVER)`` checks keep working
    against both the stub and real games."""

    NOT_STARTED = 0
    NOT_PLAYED = 0  # alias: arcengine's name for "not yet reset"
    IN_PROGRESS = 1
    NOT_FINISHED = 1  # alias: live API / arcengine name for "running"
    WIN = 2
    GAME_OVER = 3


Frame = List[List[Color]]  # row-major grid of colour indices


@dataclass
class Snapshot:
    """What RESET / ACTION returns: latest frame(s), score, state, action echo.

    `score` is the scalar reward the agent optimises. The live API and arcengine
    do NOT send a `score` field; they send `levels_completed` / `win_levels`, so
    we expose both and define `score := levels_completed` when no explicit score
    is present (the stub sets `score` directly). A WIN is `levels_completed ==
    win_levels`."""

    frames: List[Frame]
    score: float
    state: GameState
    action: Optional[GameAction] = None
    levels_completed: int = 0
    win_levels: int = 0

    @property
    def frame(self) -> Frame:
        return self.frames[-1]


# ---------------------------------------------------------------------------
# Scene functor: frame -> objects + relations + deltas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneObject:
    color: Color
    cells: Tuple[Cell, ...]

    @property
    def centroid(self) -> Cell:
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass
class Scene:
    width: int
    height: int
    objects: Tuple[SceneObject, ...]
    avatar: Optional[SceneObject]  # the object the agent's moves displace, if identifiable

    def objects_of_color(self, color: Color) -> List[SceneObject]:
        return [obj for obj in self.objects if obj.color == color]

    def colors_present(self) -> List[Color]:
        return sorted({obj.color for obj in self.objects})


def connected_components(frame: Frame, color: Color, *, diagonal: bool = False) -> List[Tuple[Cell, ...]]:
    height = len(frame)
    width = len(frame[0]) if height else 0
    seen = [[False] * width for _ in range(height)]
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    if diagonal:
        deltas += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    components: List[Tuple[Cell, ...]] = []
    for y in range(height):
        for x in range(width):
            if frame[y][x] != color or seen[y][x]:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            cells: List[Cell] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in deltas:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not seen[ny][nx] and frame[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            components.append(tuple(sorted(cells)))
    return components


def extract_scene(frame: Frame, *, avatar_color: Optional[Color] = None, diagonal: bool = False) -> Scene:
    """The scene functor. Objects are connected components of non-background
    colour. The avatar is the unique singleton of avatar_color if given,
    otherwise the smallest singleton object (a common ARC avatar heuristic;
    flagged as a heuristic, not a guarantee)."""
    height = len(frame)
    width = len(frame[0]) if height else 0
    objects: List[SceneObject] = []
    for color in range(1, NUM_COLORS):
        for cells in connected_components(frame, color, diagonal=diagonal):
            objects.append(SceneObject(color=color, cells=cells))

    avatar: Optional[SceneObject] = None
    if avatar_color is not None:
        candidates = [obj for obj in objects if obj.color == avatar_color]
        avatar = candidates[0] if candidates else None
    else:
        singletons = [obj for obj in objects if obj.size == 1]
        avatar = min(singletons, key=lambda o: (o.color, o.centroid)) if singletons else None

    return Scene(width=width, height=height, objects=tuple(objects), avatar=avatar)


# Frame-to-frame deltas: the relational change vocabulary an induction step
# would key on (moved / appeared / vanished / recoloured).

@dataclass
class SceneDelta:
    moved: Dict[Color, Tuple[Cell, Cell]] = field(default_factory=dict)
    appeared: List[Color] = field(default_factory=list)
    vanished: List[Color] = field(default_factory=list)


def scene_delta(before: Scene, after: Scene) -> SceneDelta:
    delta = SceneDelta()
    before_by_color = {obj.color: obj for obj in before.objects}
    after_by_color = {obj.color: obj for obj in after.objects}
    for color, obj in after_by_color.items():
        if color not in before_by_color:
            delta.appeared.append(color)
        elif before_by_color[color].centroid != obj.centroid:
            delta.moved[color] = (before_by_color[color].centroid, obj.centroid)
    for color in before_by_color:
        if color not in after_by_color:
            delta.vanished.append(color)
    return delta


# ---------------------------------------------------------------------------
# The correspondence: ARC colour-slot perception == cone-substrate channel
# ---------------------------------------------------------------------------

def slot_observation(scene: Scene, color: Color, mode: str = "reach", safe_radius: Optional[int] = None) -> int:
    """The cone-substrate observation for the channel bound to `color`: the
    9-way azimuth from the avatar to the nearest object of that colour, or
    HERE when the channel is "satisfied". This is exactly cone_foraging_bound's
    per-channel reading, with `color` playing the role of the channel.

    mode="reach" (seek): HERE when the avatar is on a colour object or none
    remain — "get to colour c". mode="avoid" (flee): HERE when the nearest
    colour object is at least safe_radius away or none remain — "stay away
    from colour c", the ARC analogue of the HAZARD channel."""
    if scene.avatar is None:
        return cf.ANY_OBS
    here = scene.avatar.centroid
    targets = [obj.centroid for obj in scene.objects_of_color(color) if obj is not scene.avatar]
    if not targets:
        return cf.HERE_OBS
    nearest = min(targets, key=lambda c: abs(c[0] - here[0]) + abs(c[1] - here[1]))
    distance = abs(nearest[0] - here[0]) + abs(nearest[1] - here[1])
    if mode == "avoid":
        radius = cf.SAFE_RADIUS if safe_radius is None else safe_radius
        return cf.HERE_OBS if distance >= radius else cf.azimuth_to(here, nearest)
    if nearest == here:
        return cf.HERE_OBS
    return cf.azimuth_to(here, nearest)


def scene_to_cone_level(scene: Scene, goal_color: Color) -> cf.ConeLevel:
    """Navigation projection: present an ARC scene as a foraging ConeLevel so
    the existing bound-cone machinery runs unchanged. The avatar is the start;
    objects of goal_color are 'home'. This is a faithful projection for
    navigate-to-target games and an explicit simplification for richer games
    (documented as such)."""
    if scene.avatar is None:
        raise ValueError("scene has no identifiable avatar; cannot project")
    start = scene.avatar.centroid
    goals = [obj.centroid for obj in scene.objects_of_color(goal_color) if obj is not scene.avatar]
    if not goals:
        raise ValueError(f"no objects of goal colour {goal_color}")
    home = min(goals, key=lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1]))
    return cf.ConeLevel(width=scene.width, height=scene.height, start=start, food=(), home=home)


# ---------------------------------------------------------------------------
# Offline stub environment: a tiny deterministic navigate-to-goal game
# ---------------------------------------------------------------------------

@dataclass
class StubNavigationGame:
    """A minimal ARC-shaped environment: a grid with an avatar and a goal cell.
    ACTION1-4 move the avatar; reaching the goal is a WIN. Deterministic, so it
    serves as a reproducible fixture for the connector. NOT a real ARC game —
    it exists to test that the cone perception/action wiring is correct."""

    width: int
    height: int
    avatar_color: Color
    goal_color: Color
    avatar: Cell
    goal: Cell
    state: GameState = GameState.NOT_STARTED
    steps: int = 0
    max_steps: int = 64

    @classmethod
    def random(cls, seed: int, width: int = 12, height: int = 12) -> "StubNavigationGame":
        rng = random.Random(seed)
        cells = [(x, y) for y in range(height) for x in range(width)]
        avatar, goal = rng.sample(cells, 2)
        return cls(width=width, height=height, avatar_color=4, goal_color=2, avatar=avatar, goal=goal)

    def render(self) -> Frame:
        frame = [[BACKGROUND] * self.width for _ in range(self.height)]
        gx, gy = self.goal
        frame[gy][gx] = self.goal_color
        ax, ay = self.avatar
        frame[ay][ax] = self.avatar_color
        return frame

    def snapshot(self, action: Optional[GameAction] = None) -> Snapshot:
        score = 1.0 if self.state == GameState.WIN else 0.0
        return Snapshot(frames=[self.render()], score=score, state=self.state, action=action)

    def reset(self) -> Snapshot:
        self.state = GameState.IN_PROGRESS
        self.steps = 0
        return self.snapshot()

    def step(self, action: GameAction) -> Snapshot:
        if self.state != GameState.IN_PROGRESS:
            return self.snapshot(action)
        self.steps += 1
        dx, dy = ACTION_TO_DELTA.get(action, (0, 0))
        nx, ny = self.avatar[0] + dx, self.avatar[1] + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            self.avatar = (nx, ny)
        if self.avatar == self.goal:
            self.state = GameState.WIN
        elif self.steps >= self.max_steps:
            self.state = GameState.GAME_OVER
        return self.snapshot(action)


# ---------------------------------------------------------------------------
# Live API client (network-gated; UNVERIFIED against the live service)
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://three.arcprize.org"


class ArcEnv:
    """Thin client for a live ARC-AGI-3 environment, exposing the same
    reset()/step() -> Snapshot surface as StubNavigationGame so the connector
    code above is environment-agnostic.

    Honesty notes:
    - VERIFIED live (2026-06-16/17, real key) against the documented contract at
      https://docs.arcprize.org: `X-API-Key` auth; GET /api/games (25 games, each
      {game_id, title, tags, baseline_actions}); POST /api/scorecard/open ->
      {card_id}; POST /api/cmd/RESET -> {guid, frame (list of 64x64 grids),
      state, levels_completed, win_levels, available_actions, full_reset, ...};
      POST /api/cmd/ACTION1..7 -> 200 with the same snapshot shape; POST
      /api/scorecard/close. Confirmed end to end on sb26 (clicks/keys) and ls20
      (directional, win_levels=7).
    - Importing this module never touches the network. A request happens only
      when a method is called on an ArcEnv instance, which requires an API key,
      so the offline tests stay hermetic.

    Verified contract details (docs.arcprize.org + live observation):
    - NO `score` FIELD. Progress is `levels_completed` out of `win_levels`; a
      WIN is levels_completed == win_levels. _snapshot_from_payload defines
      Snapshot.score := levels_completed when no explicit score is present, so
      the reward signal is live (the previous payload.get("score") was always
      0.0 and silently dead).
    - `state` is reported by NAME and is "NOT_FINISHED" while running (NOT
      "IN_PROGRESS"); GameState aliases NOT_FINISHED/NOT_PLAYED so the mapping
      and downstream comparisons are exact rather than relying on a fallback.
    - ACTION7 exists (undo, where the game offers it); the action set is
      ACTION1-5 simple, ACTION6 coordinate (x,y), ACTION7 undo.
    - `game_id` must be the FULL id ("ls20-9607627b") for the action loop. RESET
      accepts the bare short code ("ls20") but the next ACTION then fails with
      "game ls20 not found". reset() resolves a short code to the full id via
      /api/games (resolve_full_game_id) before issuing RESET.
    - COOKIES ARE LOAD-BEARING. The RESET response sets a session-affinity
      cookie (AWSALB*) that MUST be sent on every following ACTION, or the action
      hits a backend without the game and returns "game not found". This client
      uses a cookie-aware opener for the same effect.
    - The action payload is {game_id, guid} (+ x,y for ACTION6); card_id is a
      RESET/scorecard field, not an action field.
    - `frame` is a LIST of 64x64 grids (the environment may settle over several
      internal frames); the latest is frames[-1].
    - For key-free, rate-limit-free LOCAL play of the same games, see
      LocalArcEnv below (arc_agi toolkit, OperationMode.OFFLINE/NORMAL).
    """

    def __init__(
        self,
        game_id: str,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        reset_retries: int = 10,
    ) -> None:
        self.game_id = normalize_game_id(game_id)
        self.api_key = api_key or os.environ.get("ARC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ARC-AGI-3 API key required: pass api_key= or set ARC_API_KEY. "
                "Get one at https://three.arcprize.org (offline work needs no key)."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.reset_retries = reset_retries
        self.card_id: Optional[str] = None
        self.guid: Optional[str] = None
        self.available_actions: List[int] = []
        # The official toolkit (arc_agi.remote_wrapper) uses a requests.Session
        # with a shared cookie jar: the RESET response sets a session-affinity
        # cookie that MUST be sent on every following ACTION, or the action hits
        # a backend without the game and returns "game not found". urllib needs
        # an explicit cookie-aware opener for the same effect.
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def _request(self, method: str, path: str, payload: Optional[Dict] = None) -> Tuple[int, object]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else (b"" if method == "POST" else None)
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            except Exception:
                return exc.code, None

    @classmethod
    def list_games(cls, api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> List[Dict]:
        key = api_key or os.environ.get("ARC_API_KEY")
        if not key:
            raise RuntimeError("ARC-AGI-3 API key required to list games")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/games",
            headers={"X-API-Key": key, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def open_scorecard(self) -> str:
        status, payload = self._request("POST", "/api/scorecard/open", {})
        if status != 200 or not isinstance(payload, dict) or "card_id" not in payload:
            raise RuntimeError(f"scorecard open failed: {status} {payload}")
        self.card_id = payload["card_id"]
        return self.card_id

    def close_scorecard(self) -> None:
        if self.card_id:
            self._request("POST", "/api/scorecard/close", {"card_id": self.card_id})
            self.card_id = None

    def _snapshot_from_payload(self, payload: Dict, action: Optional[GameAction] = None) -> Snapshot:
        raw = payload.get("frame") or payload.get("frames") or []
        # `frame` is a list of 2D grids; normalize to a list of grids.
        if raw and isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list):
            frames = [list(g) for g in raw]
        elif raw and isinstance(raw[0], list):
            frames = [raw]
        else:
            frames = [[[BACKGROUND]]]
        self.available_actions = list(payload.get("available_actions", self.available_actions))
        state_name = str(payload.get("state", "NOT_FINISHED")).upper()
        state = GameState[state_name] if state_name in GameState.__members__ else GameState.IN_PROGRESS
        # The live API/arcengine carry no `score`; progress is levels_completed
        # out of win_levels. Fall back to levels_completed so reward is live.
        levels = int(payload.get("levels_completed", 0) or 0)
        win = int(payload.get("win_levels", 0) or 0)
        raw_score = payload.get("score")
        score = float(raw_score) if raw_score is not None else float(levels)
        return Snapshot(frames=frames, score=score, state=state, action=action,
                        levels_completed=levels, win_levels=win)

    def _ensure_full_game_id(self) -> None:
        """Resolve a bare short code to the full id once, via /api/games. The
        full id is required for the action loop (see normalize_game_id)."""
        if "-" in self.game_id:
            return
        try:
            self.game_id = resolve_full_game_id(
                self.game_id, self.list_games(self.api_key, self.base_url))
        except Exception:
            pass  # leave the short code; RESET may still work for diagnostics

    def reset(self) -> Snapshot:
        self._ensure_full_game_id()
        if self.card_id is None:
            self.open_scorecard()
        last: object = None
        for _ in range(self.reset_retries):
            status, payload = self._request("POST", "/api/cmd/RESET", {"game_id": self.game_id, "card_id": self.card_id})
            if status == 200 and isinstance(payload, dict):
                self.guid = payload.get("guid", self.guid)
                return self._snapshot_from_payload(payload)
            last = payload
            # Eventual consistency: "game not found" for ~5 calls / ~15s while
            # the instance provisions; a constant ~3s poll reliably catches it.
            time.sleep(3.0)
        raise RuntimeError(f"RESET failed after {self.reset_retries} tries: {last}")

    def step(self, action: GameAction, x: Optional[int] = None, y: Optional[int] = None,
             retries: int = 5) -> Snapshot:
        # Documented action body is {game_id, guid} (+ x,y for ACTION6); card_id
        # is a RESET/scorecard field, not an action field.
        body: Dict[str, object] = {"game_id": self.game_id, "guid": self.guid}
        if action == GameAction.ACTION6:
            body["x"], body["y"] = x or 0, y or 0
        last: object = None
        for _ in range(retries):
            status, payload = self._request("POST", f"/api/cmd/{action.name}", body)
            if status == 200 and isinstance(payload, dict):
                return self._snapshot_from_payload(payload, action)
            # The post-RESET provisioning window can still report "not found"
            # for the first action(s); a 400 means the action did not apply, so
            # retrying is safe (no double-application).
            last = payload
            time.sleep(3.0)
        raise RuntimeError(f"{action.name} failed after {retries} tries: {last}")


def normalize_game_id(game_id: str) -> str:
    """Lowercase/trim a game id, PRESERVING the version suffix. The full id
    ("ls20-9607627b") is required for the RESET+ACTION loop: RESET accepts the
    bare short code ("ls20") but the following ACTION then fails with "game
    <code> not found", because the action is routed by the full id. Use
    resolve_full_game_id() to turn a short code into the full id."""
    return game_id.strip().lower()


def resolve_full_game_id(game_id: str, games: Sequence[Dict]) -> str:
    """Map a (possibly short) game id to the full id ("ls20-9607627b") whose
    short code matches, using the /api/games listing. Returns the input
    unchanged if it is already a full id or no match is found. Pure function —
    the network (list_games) is supplied by the caller, so this is testable
    offline."""
    gid = normalize_game_id(game_id)
    if "-" in gid:
        return gid
    for game in games:
        full = str(game.get("game_id", ""))
        if full.split("-", 1)[0].lower() == gid:
            return full.lower()
    return gid


# ---------------------------------------------------------------------------
# Local API client (offline, via the official arc_agi toolkit / arcengine)
# ---------------------------------------------------------------------------

DEFAULT_ENVIRONMENTS_DIR = "environment_files"


class LocalArcEnv:
    """LOCAL play of the same ARC-AGI-3 games through the official `arc_agi`
    toolkit (the `arcengine` runtime), exposing the SAME reset()/step()->Snapshot
    surface as ArcEnv. Every connector below (run_seek_leg_on_game,
    ArcEnvironment) therefore runs unchanged against a local game — but with NO
    per-step network and no rate limits (~2000 FPS, per docs.arcprize.org
    local-vs-online).

    operation_mode (docs.arcprize.org/local-vs-online):
      - "offline": local only. The game files must already be under
        environments_dir (download them once with mode="normal", or the toolkit
        CLI). No API key and no network are used at all.
      - "normal": the first time, download the game source from the API (needs a
        key), cache it under environments_dir, then run it LOCALLY; later runs
        can be "offline". This is the default because it "just works" given a key
        and is fully local after the one-time fetch.

    Honesty notes:
    - VERIFIED locally (2026-06-16/17): Arcade(NORMAL).make("ls20") downloads
      ls20.py + metadata.json into environments_dir and runs it via
      LocalEnvironmentWrapper; reset()/step() return arcengine FrameDataRaw with
      the same fields as the REST API (frame list, state="NOT_FINISHED",
      levels_completed, win_levels, available_actions). On ls20 the directional
      actions [1,2,3,4] move the world locally exactly as online.
    - LAZY IMPORT: arc_agi/arcengine are imported only when an instance is
      constructed. Importing this module never requires them, so the hermetic
      offline test-suite (which constructs only the stub / ArcEnv-with-dummy-key)
      stays dependency-free.
    """

    def __init__(
        self,
        game_id: str,
        operation_mode: str = "normal",
        environments_dir: str = DEFAULT_ENVIRONMENTS_DIR,
        api_key: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        from arc_agi import Arcade, OperationMode  # lazy; only needed for local play

        mode = OperationMode(operation_mode) if isinstance(operation_mode, str) else operation_mode
        key = api_key or os.environ.get("ARC_API_KEY", "")
        needs_key = mode in (OperationMode.NORMAL, OperationMode.ONLINE, OperationMode.COMPETITION)
        if needs_key and not key:
            raise RuntimeError(
                f"operation_mode={mode.value!r} needs an API key to fetch the game once "
                "(pass api_key= or set ARC_API_KEY). For fully key-free play of games "
                "already downloaded under environments_dir, use operation_mode='offline'."
            )
        self.game_id = normalize_game_id(game_id)  # toolkit accepts short or full id
        self.seed = seed
        self.available_actions: List[int] = []
        self._arcade = Arcade(arc_api_key=key, operation_mode=mode, environments_dir=environments_dir)
        self._env = None  # arc_agi EnvironmentWrapper, created on reset()

    def _make(self):
        env = self._arcade.make(self.game_id, seed=self.seed)
        if env is None:
            raise RuntimeError(
                f"could not make local game {self.game_id!r}. In 'offline' mode the game "
                f"files must already exist under the environments_dir; download them once "
                f"with operation_mode='normal' and a key."
            )
        return env

    def _snapshot_from_frame(self, fd, action: Optional[GameAction] = None) -> Snapshot:
        """arcengine FrameDataRaw -> Snapshot (same shape as ArcEnv produces)."""
        import numpy as np  # arcengine dependency; lazy

        arr = np.asarray(fd.frame)
        if arr.ndim == 2:  # a single grid
            frames = [arr.astype(int).tolist()]
        else:              # list of settling grids (G, H, W)
            frames = [arr[i].astype(int).tolist() for i in range(arr.shape[0])]
        state_obj = fd.state
        state_name = getattr(state_obj, "name", str(state_obj).split(".")[-1]).upper()
        state = GameState[state_name] if state_name in GameState.__members__ else GameState.IN_PROGRESS
        self.available_actions = list(fd.available_actions or self.available_actions)
        levels = int(fd.levels_completed or 0)
        win = int(fd.win_levels or 0)
        return Snapshot(frames=frames, score=float(levels), state=state, action=action,
                        levels_completed=levels, win_levels=win)

    def reset(self) -> Snapshot:
        self._env = self._make()
        fd = self._env.reset()
        if fd is None:
            raise RuntimeError(f"local RESET failed for {self.game_id!r}")
        return self._snapshot_from_frame(fd)

    def step(self, action: GameAction, x: Optional[int] = None, y: Optional[int] = None) -> Snapshot:
        from arcengine import GameAction as EngineAction  # lazy

        assert self._env is not None, "call reset() first"
        # arcengine's GameAction disallows value-based construction; map by name
        # (our GameAction names mirror arcengine's: ACTION1..7).
        engine_action = EngineAction[action.name]
        data = {"x": int(x or 0), "y": int(y or 0)} if action == GameAction.ACTION6 else None
        fd = self._env.step(engine_action, data=data)
        if fd is None:
            raise RuntimeError(f"local step {action.name} failed for {self.game_id!r}")
        return self._snapshot_from_frame(fd, action)

    def close_scorecard(self) -> None:
        """Match ArcEnv's surface. Local play keeps a scorecard inside the
        Arcade; close it best-effort so ArcEnvironment.close() is uniform."""
        try:
            self._arcade.close_scorecard()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ArcEnvironment: the ARC connector for the general method (cone_method).
# It PROVIDES the actions (Section 14). The method consumes it abstractly.
# ---------------------------------------------------------------------------

class ArcEnvironment:
    """ARC-AGI-3 as a `cone_method.Environment`. This is the SPECIFIC half: it
    provides the action primitives (simple ACTIONs present in the game, plus
    perception-derived ACTION6 clicks at candidate object cells), the
    perception (frame -> per-colour features), and the reward (game score).
    The general method never names these.

    Honest limitation (validated live): ARC clicks only register on hidden
    `sys_click` sprites (per arcengine), and the API exposes no list of valid
    click targets. The click candidates here (small-component cells) are a
    heuristic; on the games tested they did not coincide with the hidden
    clickable sprites, so generic actions did not move the world. Discovering
    the right action primitives per game is the open problem, and it lives
    here in the connector, not in the method."""

    def __init__(self, game_id: str, max_click_candidates: int = 24, **env_kwargs) -> None:
        self.env = ArcEnv(game_id, **env_kwargs)
        self.max_click_candidates = max_click_candidates
        self._snap: Optional[Snapshot] = None

    def reset(self) -> None:
        self._snap = self.env.reset()

    def _frame(self) -> Frame:
        assert self._snap is not None, "call reset() first"
        return self._snap.frame

    def actions(self) -> List[Tuple]:
        """Connector-provided action primitives for the current state."""
        avail = self.env.available_actions or [1, 2, 3, 4, 5]
        # Simple actions are everything except the coordinate click (6): the
        # directional/interact keys 1-5 plus 7 (undo, where offered).
        acts: List[Tuple] = [("KEY", n) for n in avail if n != 6 and 1 <= n <= 7]
        if 6 in avail:
            frame = self._frame()
            cells = []
            for color in range(1, NUM_COLORS):
                for comp in connected_components(frame, color):
                    if 1 <= len(comp) <= 60:  # small components = candidate widgets
                        cx = sorted(c[0] for c in comp)[len(comp) // 2]
                        cy = sorted(c[1] for c in comp)[len(comp) // 2]
                        cell = min(comp, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                        cells.append(("CLICK", cell[0], cell[1]))
            acts.extend(cells[: self.max_click_candidates])
        return acts

    def apply(self, action: Tuple) -> None:
        kind = action[0]
        if kind == "KEY":
            self._snap = self.env.step(GameAction(action[1]))
        elif kind == "CLICK":
            self._snap = self.env.step(GameAction.ACTION6, x=action[1], y=action[2])
        else:
            raise ValueError(f"unknown action {action!r}")

    def features(self) -> Dict[str, float]:
        """Perception: per-colour scene features from the current frame."""
        frame = self._frame()
        feats: Dict[str, float] = {}
        for color in range(1, NUM_COLORS):
            n = len(connected_components(frame, color))
            if n:
                feats[f"present@{color}"] = 1.0
                feats[f"count@{color}"] = min(1.0, n / 16.0)
        return feats

    def reward(self) -> float:
        return self._snap.score if self._snap else 0.0

    def done(self) -> bool:
        return bool(self._snap and self._snap.state in (GameState.WIN, GameState.GAME_OVER))

    def close(self) -> None:
        self.env.close_scorecard()


# ---------------------------------------------------------------------------
# Channel-blind seek leg driven directly by scene perception
# ---------------------------------------------------------------------------

def run_seek_leg_on_game(
    game,
    leg: cf.Leg,
    goal_color: Color,
    avatar_color: Optional[Color] = None,
    max_steps: int = 64,
) -> Snapshot:
    """Drive any reset()/step()->Snapshot environment with a channel-blind seek
    leg bound to goal_color. Works on StubNavigationGame and (untested) ArcEnv
    alike, since both expose the same surface.

    This exercises the full connector: at each turn the scene functor extracts
    the avatar and objects, the colour-slot observation feeds the leg (whose
    rules never see the colour), and the leg's move maps to a GameAction. It is
    the ARC analogue of CALL(seek, goal_color) — a priced binding of an
    abstract leg to a perceptual slot."""
    if avatar_color is None:
        avatar_color = getattr(game, "avatar_color", None)
    snapshot = game.reset()
    leg_map = leg.genome.rule_map()
    substate = 0
    for _ in range(max_steps):
        if snapshot.state != GameState.IN_PROGRESS:
            break
        scene = extract_scene(snapshot.frame, avatar_color=avatar_color)
        obs = slot_observation(scene, goal_color)
        rule = leg_map.get((substate, obs)) or leg_map.get((substate, cf.ANY_OBS))
        if rule is None:
            break  # sparse-halt, exactly like every machine in the repository
        for action in rule.actions:
            if cf.is_move(action):
                snapshot = game.step(MOVE_TO_ACTION[action])
            elif action == cf.RETURN_ACTION:
                return snapshot
            if snapshot.state != GameState.IN_PROGRESS:
                return snapshot
        substate = rule.next_state
    return snapshot
