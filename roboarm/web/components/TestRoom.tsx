"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  loadCampaignBundle,
  replayMoments,
  type CampaignAttempt,
  type CampaignBundle,
  type CampaignLineageProfile,
  type EvidenceMode,
} from "@/lib/campaign";
import {
  ACTION_LABELS,
  type Action,
  type PublicTelemetry,
  type WorldSnapshot,
} from "@/lib/model";

const RobotScene = dynamic(() => import("./RobotScene"), {
  ssr: false,
  loading: () => (
    <div className="scene-loading">
      <span className="spinner" />
      Calibrating replay camera…
    </div>
  ),
});

const RUN_STEP_MS = 190;

function degrees(radians: number): string {
  return `${((radians * 180) / Math.PI).toFixed(1)}°`;
}

function shortHash(value: string | undefined): string {
  return value === undefined ? "—" : value.slice(0, 12);
}

function eventTitle(kind: string): string {
  return kind
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function RgbObservation({
  frame,
  turn,
  frameHash,
  mechanicsTest,
}: {
  frame: Uint8Array;
  turn: number;
  frameHash: string;
  mechanicsTest: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (context === null || context === undefined) return;
    const image = context.createImageData(128, 72);
    for (let pixel = 0; pixel < 128 * 72; pixel += 1) {
      image.data[pixel * 4] = frame[pixel * 3];
      image.data[pixel * 4 + 1] = frame[pixel * 3 + 1];
      image.data[pixel * 4 + 2] = frame[pixel * 3 + 2];
      image.data[pixel * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [frame]);

  return (
    <div className="observation-card">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">
            {mechanicsTest
              ? "EXACT TEST CAMERA INPUT"
              : "EXACT GODEL-KOLMOGOROV MACHINE INPUT"}
          </span>
          <h2>Recorded RGB camera</h2>
        </div>
        <span className="micro-badge">TURN {String(turn).padStart(2, "0")}</span>
      </div>
      <div className="observation-frame">
        <canvas
          ref={canvasRef}
          width={128}
          height={72}
          data-testid="rgb-observation"
          aria-label="Exact processed 128 by 72 Logitech C920s RGB observation"
        />
        <span className="corner corner-tl" />
        <span className="corner corner-tr" />
        <span className="corner corner-bl" />
        <span className="corner corner-br" />
      </div>
      <div className="observation-meta">
        <span>128 × 72 × 3 RGB8</span>
        <span>C920s · 78° DIAGONAL</span>
        <span>SHA {shortHash(frameHash)}</span>
      </div>
      <p className="observation-disclaimer">
        {mechanicsTest
          ? "Exact deterministic mechanics-test camera frame."
          : "These processed C920s bytes and the nearest host-timestamped RoArm feedback packet below were supplied to the Godel-Kolmogorov machine."}
      </p>
    </div>
  );
}

function Telemetry({ telemetry }: { telemetry: PublicTelemetry }) {
  const command = telemetry.controller.command_json;
  const feedback = telemetry.arm.feedback;
  const selected = telemetry.controller.selected_coordinate;
  const azimuth = Math.atan2(command.y, command.x);
  const reachMm = Math.hypot(command.x, command.y);
  const controllerState = telemetry.controller.interlocked
    ? "HOST INTERLOCK · NOT TRANSMITTED"
    : telemetry.sample.sequence === 0
      ? "INITIAL SENSOR BOUNDARY"
      : "COMMAND ACCEPTED";

  return (
    <section className="telemetry-panel" aria-label="Recorded arm telemetry">
      <div className="panel-heading telemetry-heading">
        <div>
          <span className="eyebrow">EXACT CONTROLLER INPUT</span>
          <h2>Command / measured</h2>
        </div>
        <div className="state-bus">
          <span className="state-dot" />
          GODEL-KOLMOGOROV MACHINE INPUT
        </div>
      </div>

      <div className="telemetry-grid">
        <div className="telemetry-block">
          <span className="metric-label">COMMAND SPACE</span>
          <dl className="metric-list">
            <div className={selected === "azimuth" ? "selected" : ""}>
              <dt>Azimuth</dt>
              <dd>{degrees(azimuth)}</dd>
            </div>
            <div className={selected === "reach" ? "selected" : ""}>
              <dt>Reach</dt>
              <dd>{Math.round(reachMm)} mm</dd>
            </div>
            <div className={selected === "height" ? "selected" : ""}>
              <dt>Height</dt>
              <dd>{Math.round(command.z)} mm</dd>
            </div>
          </dl>
        </div>

        <div className="telemetry-block">
          <span className="metric-label">ACTUAL TCP · MM</span>
          <dl className="metric-list compact">
            <div><dt>X</dt><dd>{feedback.x >= 0 ? "+" : ""}{feedback.x.toFixed(1)}</dd></div>
            <div><dt>Y</dt><dd>{feedback.y >= 0 ? "+" : ""}{feedback.y.toFixed(1)}</dd></div>
            <div><dt>Z</dt><dd>{feedback.z >= 0 ? "+" : ""}{feedback.z.toFixed(1)}</dd></div>
          </dl>
        </div>

        <div className="telemetry-block joints-block">
          <span className="metric-label">JOINT ENCODERS</span>
          <dl className="metric-list compact">
            {[feedback.b, feedback.s, feedback.e, feedback.t].map((joint, index) => (
              <div key={index}>
                <dt>Q{index}</dt>
                <dd>{degrees(joint)}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="telemetry-block status-block">
          <span className="metric-label">RAW SERVO LOAD</span>
          <div className="load-row">
            <div className="load-track">
              <span style={{ width: `${Math.max(2, Math.min(100, Math.abs(feedback.torH) / 10.23))}%` }} />
            </div>
            <strong>H {feedback.torH}</strong>
          </div>
          <div className="object-state">
            <span className={telemetry.controller.interlocked ? "" : "attached"} />
            {controllerState}
          </div>
          <div className="gripper-state">
            HAND {degrees(feedback.t)} · {feedback.v / 100} V · SKEW {telemetry.sample.sensor_skew_ms.toFixed(1)} ms
          </div>
        </div>
      </div>
    </section>
  );
}

function EventTrace({ snapshot }: { snapshot: WorldSnapshot }) {
  const visibleEvents = snapshot.eventLog.slice(-7).reverse();
  return (
    <section className="event-panel" aria-label="Recorded mechanics event trace">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">REPLAY MECHANICS</span>
          <h2>Contact & state trace</h2>
        </div>
        <span className="event-count">{snapshot.eventLog.length}</span>
      </div>
      <div className="event-list" data-testid="event-trace">
        {visibleEvents.length === 0 ? (
          <div className="event-empty">No mechanics event at this boundary.</div>
        ) : (
          visibleEvents.map((event, index) => (
            <div
              className={`event-row event-${event.kind}`}
              key={`${event.turn}-${event.kind}-${index}`}
            >
              <span className="event-turn">T{String(event.turn).padStart(2, "0")}</span>
              <span className="event-pip" />
              <span className="event-copy">
                <strong>{eventTitle(event.kind)}</strong>
                <small>{event.detail}</small>
              </span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function EvidencePanel({
  bundle,
  selectedIndex,
  attempt,
  cursor,
  onSelect,
  onSeek,
  mechanicsTest,
}: {
  bundle: CampaignBundle;
  selectedIndex: number;
  attempt: CampaignAttempt;
  cursor: number;
  onSelect: (index: number) => void;
  onSeek: (index: number) => void;
  mechanicsTest: boolean;
}) {
  return (
    <section
      className="evidence-panel"
      aria-label={
        mechanicsTest
          ? "Developer mechanics fixtures"
          : "Godel-Kolmogorov machine campaign evidence"
      }
    >
      <div className="panel-heading">
        <div>
          <span className="eyebrow">
            {mechanicsTest ? "TEST-ONLY REPLAY EVIDENCE" : "LLM PROPOSER EVIDENCE"}
          </span>
          <h2>
            {mechanicsTest ? "Fixtures & public actions" : "Attempts & public actions"}
          </h2>
        </div>
        <span className="micro-badge">
          {mechanicsTest
            ? "NOT MACHINE LEARNING EVIDENCE"
            : "NO BROWSER SOLVER"}
        </span>
      </div>
      <div
        className="attempt-tabs"
        role="tablist"
        aria-label={mechanicsTest ? "Mechanics fixtures" : "Campaign attempts"}
      >
        {bundle.attempts.map((candidate, index) => (
          <button
            type="button"
            role="tab"
            aria-selected={selectedIndex === index}
            className={`attempt-tab attempt-${candidate.disposition}`}
            key={`${candidate.disposition}-${index}`}
            onClick={() => onSelect(index)}
            data-testid={`attempt-${index}`}
            data-disposition={candidate.disposition}
            data-replay-stage={candidate.replay_stage ?? ""}
          >
            <span>
              {candidate.disposition === "failed" ||
              candidate.disposition === "expected-rejection"
                ? "×"
                : "✓"}
            </span>
            <div>
              <strong>
                {candidate.disposition === "failed"
                  ? `Failed operation${
                      candidate.generation === undefined
                        ? ""
                        : ` · G${candidate.generation}`
                    }`
                  : candidate.disposition === "promoted"
                    ? candidate.replay_stage === "discovery_commit"
                      ? "Safety-authorized success"
                      : "Promoted exact replay"
                    : candidate.disposition === "expected-rejection"
                      ? "Collision regression"
                      : "Completed mechanics replay"}
              </strong>
              <small>
                {candidate.actions.length} actions ·{" "}
                {candidate.scenario_id ?? candidate.trace_role}
              </small>
            </div>
          </button>
        ))}
      </div>
      <div className="evidence-receipts">
        <span>
          {mechanicsTest ? "FIXTURE SET" : "CAMPAIGN"}{" "}
          <strong>{bundle.manifest.campaign_id}</strong>
        </span>
        <span>
          {mechanicsTest ? "FIXTURE" : "SOURCE"}{" "}
          <strong>
            {mechanicsTest
              ? attempt.fixture_id ?? "—"
              : shortHash(
                  attempt.source_tree_sha256 ??
                    attempt.fsa_receipt_sha256,
                )}
          </strong>
        </span>
        <span>REPLAY <strong>{shortHash(attempt.replay_receipt_sha256)}</strong></span>
      </div>
      <div className="action-timeline" data-testid="action-timeline">
        <button
          type="button"
          className={cursor === 0 ? "timeline-row active" : "timeline-row"}
          onClick={() => onSeek(0)}
        >
          <span>T00</span>
          <strong>Initial observation</strong>
          <small>reward 0</small>
        </button>
        {attempt.steps.map((step, index) => {
          const selected = cursor === index + 1;
          const rejected = step.visual_state.robot.rejected;
          return (
            <button
              type="button"
              className={[
                "timeline-row",
                selected ? "active" : "",
                rejected ? "rejected" : "",
                step.levels_completed > 0 ? "rewarded" : "",
              ].join(" ")}
              key={`${index}-${step.frame_sha256}`}
              onClick={() => onSeek(index + 1)}
            >
              <span>T{String(index + 1).padStart(2, "0")}</span>
              <strong>
                A{step.action} · {ACTION_LABELS[step.action]}
              </strong>
              <small>
                {step.levels_completed > 0
                  ? `reward ${step.levels_completed}`
                  : rejected
                    ? "rejected"
                    : "reward 0"}
              </small>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function MetricTrace({
  label,
  values,
  unit,
  accent,
}: {
  label: string;
  values: number[];
  unit: string;
  accent: "cyan" | "orange";
}) {
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = Math.max(1, maximum - minimum);
  const points = values.map((value, index) => ({
    x: values.length === 1 ? 230 : 20 + (index * 420) / (values.length - 1),
    y: 16 + ((maximum - value) * 62) / span,
    value,
  }));
  return (
    <div className={`lineage-chart lineage-chart-${accent}`}>
      <div className="lineage-chart-heading">
        <strong>{label}</strong>
        <span>{unit}</span>
      </div>
      <svg viewBox="0 0 460 110" role="img" aria-label={`${label}: ${values.join(", ")}`}>
        <path className="lineage-gridline" d="M20 16H440 M20 47H440 M20 78H440" />
        <polyline points={points.map((point) => `${point.x},${point.y}`).join(" ")} />
        {points.map((point, index) => (
          <g key={`${label}-${index}`}>
            <circle cx={point.x} cy={point.y} r="4" />
            <text x={point.x} y={point.y - 9} textAnchor="middle">{point.value}</text>
            <text className="lineage-generation-label" x={point.x} y="101" textAnchor="middle">G{index + 1}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function LineagePanel({ profile }: { profile: CampaignLineageProfile }) {
  const generations = profile.generations;
  const interpretation = profile.interpretation;
  const finalLegs = generations.at(-1)?.transitive_unchanged_called_legs ?? [];
  return (
    <section
      className="lineage-panel"
      aria-label="Retained legs and marginal complexity"
      data-testid="lineage-panel"
      data-profile-kind={profile.profile_kind}
    >
      <div className="panel-heading lineage-heading">
        <div>
          <span className="eyebrow">AUDITED RETAINED-SOURCE LINEAGE</span>
          <h2>Retained legs & marginal complexity</h2>
        </div>
        <div className="lineage-badges">
          <span className="micro-badge">CONSTRUCTION PROFILE</span>
          <span className="micro-badge lineage-caveat">NOT SOLVED-LEVEL SAWTOOTH</span>
        </div>
      </div>

      <div className="lineage-summary">
        <p>{interpretation.reason} The two coordinates below use independent scales.</p>
        <dl>
          <div><dt>NET REVERSALS</dt><dd>{interpretation.historical_net_growth_direction_changes}</dd></div>
          <div><dt>TRANSITIVE REUSE</dt><dd>{interpretation.transitive_reuse_generations}/{generations.length}</dd></div>
          <div><dt>STRICT DIRECT</dt><dd>{interpretation.direct_reuse_generations}</dd></div>
          <div><dt>SHARP + DIRECT</dt><dd>{interpretation.sharp_direct_coupled_witnesses}</dd></div>
        </dl>
      </div>

      <div className="lineage-chart-grid">
        <MetricTrace
          label="Historical net-growth C"
          values={generations.map((row) => row.historical_net_growth)}
          unit="positive net description growth"
          accent="orange"
        />
        <MetricTrace
          label="Conditional AST marginal M"
          values={generations.map((row) => row.conditional_ast_zlib_bytes)}
          unit="zlib-9 bytes of novel AST units"
          accent="cyan"
        />
      </div>

      <div className="lineage-generation-grid">
        {generations.map((row) => (
          <article className={row.winning_checkpoint ? "lineage-generation winning" : "lineage-generation"} key={row.generation}>
            <div><strong>G{row.generation}</strong><span>{row.winning_checkpoint ? "WINNING CHECKPOINT" : "ADMITTED REVISION"}</span></div>
            <p>{row.milestone}</p>
            <dl>
              <div><dt>C</dt><dd>{row.historical_net_growth}</dd></div>
              <div><dt>M</dt><dd>{row.conditional_ast_zlib_bytes} B</dd></div>
              <div><dt>REUSED AST</dt><dd>{row.literal_reused_top_level_nodes}</dd></div>
              <div><dt>INVOKED LEGS</dt><dd>{row.transitive_unchanged_called_legs.length}</dd></div>
            </dl>
          </article>
        ))}
      </div>

      <div className="lineage-legs">
        <div>
          <span className="eyebrow">FINAL GENERATION · UNCHANGED AND REACHABLE</span>
          <strong>{finalLegs.length} transitively invoked retained legs</strong>
        </div>
        <div className="lineage-leg-list">
          {finalLegs.map((leg) => <code key={leg}>{leg.replace("legs.py:", "")}</code>)}
        </div>
      </div>
      <p className="lineage-footnote">
        Direct unchanged calls from <code>propose_level_1</code> are zero: each player enters through a newly named composition. Reuse shown here is real but transitive; the final AST drop is not the predeclared half-or-more “sharp” threshold.
      </p>
    </section>
  );
}

function phaseLabel(
  attempt: CampaignAttempt,
  cursor: number,
  snapshot: WorldSnapshot,
): string {
  if (snapshot.success) return "Sparse reward acquired · placement verified";
  if (snapshot.robot.rejected) {
    return `Command rejected · ${snapshot.robot.rejectionReason}`;
  }
  const latest = snapshot.events.at(-1);
  if (latest !== undefined) return eventTitle(latest.kind);
  if (cursor === 0) return "Initial public observation";
  if (cursor === attempt.actions.length && attempt.disposition === "failed") {
    return "Attempt ended without sparse reward";
  }
  return `${ACTION_LABELS[attempt.actions[cursor - 1]]} · reward remains 0`;
}

function ViewerUnavailable({
  error,
  mechanicsTest,
}: {
  error: string | null;
  mechanicsTest: boolean;
}) {
  return (
    <main className="test-room viewer-unavailable" data-status="unavailable">
      <section>
        <span className="brand-kicker">
          {mechanicsTest
            ? "GODEL-KOLMOGOROV MACHINE · DEVELOPER TEST"
            : "GODEL-KOLMOGOROV MACHINE · REPLAY VIEWER"}
        </span>
        <h1>
          {error === null
            ? mechanicsTest
              ? "Loading mechanics fixtures…"
              : "Loading campaign evidence…"
            : mechanicsTest
              ? "No mechanics fixture available"
              : "No admitted replay yet"}
        </h1>
        <p>
          {error ??
            (mechanicsTest
              ? "Reading Python-authoritative mechanics fixtures."
              : "Reading replay-validated failure and promotion artifacts from the local campaign export.")}
        </p>
        <small>
          {mechanicsTest
            ? "Developer regression only. This is not proposer activity, discovery, learning, or promotion evidence."
            : "This page never substitutes the canonical mechanics fixture for a headless-Codex Godel-Kolmogorov machine campaign."}
        </small>
      </section>
    </main>
  );
}

export default function TestRoom({
  basePath = "/campaign",
  evidenceMode = "gkm",
}: {
  basePath?: string;
  evidenceMode?: EvidenceMode;
}) {
  const mechanicsTest = evidenceMode === "mechanics-test";
  const [bundle, setBundle] = useState<CampaignBundle | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [cursor, setCursor] = useState(0);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let active = true;
    loadCampaignBundle(basePath, evidenceMode)
      .then((loaded) => {
        if (active) setBundle(loaded);
      })
      .catch((error: unknown) => {
        if (active) {
          setLoadError(error instanceof Error ? error.message : String(error));
        }
      });
    return () => {
      active = false;
    };
  }, [basePath, evidenceMode]);

  const attempt = bundle?.attempts[selectedIndex] ?? null;
  const moments = useMemo(
    () => (attempt === null ? [] : replayMoments(attempt)),
    [attempt],
  );
  const moment = moments[cursor];

  const selectAttempt = useCallback((index: number) => {
    setRunning(false);
    setSelectedIndex(index);
    setCursor(0);
  }, []);

  const seek = useCallback(
    (index: number) => {
      setRunning(false);
      setCursor(Math.max(0, Math.min(index, moments.length - 1)));
    },
    [moments.length],
  );

  const startOrResume = useCallback(() => {
    if (moments.length === 0) return;
    if (cursor >= moments.length - 1) setCursor(0);
    setRunning(true);
  }, [cursor, moments.length]);

  useEffect(() => {
    if (!running) return;
    if (cursor >= moments.length - 1) {
      setRunning(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setCursor((value) => value + 1);
    }, RUN_STEP_MS);
    return () => window.clearTimeout(timer);
  }, [cursor, moments.length, running]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.code === "Space") {
        event.preventDefault();
        if (running) setRunning(false);
        else startOrResume();
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        seek(cursor + 1);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        seek(cursor - 1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [cursor, running, seek, startOrResume]);

  if (bundle === null || attempt === null || moment === undefined) {
    return <ViewerUnavailable error={loadError} mechanicsTest={mechanicsTest} />;
  }

  const { snapshot } = moment;
  const atEnd = cursor === moments.length - 1;
  const failed =
    atEnd &&
    (attempt.disposition === "failed" ||
      attempt.disposition === "expected-rejection");
  const status = snapshot.success
    ? "success"
    : failed
      ? "failed"
      : running
        ? "running"
        : "idle";
  const progress =
    attempt.actions.length === 0 ? 0 : (cursor / attempt.actions.length) * 100;

  return (
    <main
      className={`test-room status-${status}`}
      data-status={status}
      data-turn={snapshot.turn}
      data-success={snapshot.success ? "true" : "false"}
      data-disposition={attempt.disposition}
      data-attempt-index={selectedIndex}
      data-replay-stage={attempt.replay_stage ?? ""}
      data-campaign={bundle.manifest.campaign_id}
      data-evidence-kind={bundle.manifest.export_kind}
    >
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <span className="brand-kicker">
              {mechanicsTest
                ? "GODEL-KOLMOGOROV MACHINE · DEVELOPER REGRESSION"
                : "GODEL-KOLMOGOROV MACHINE · CAMPAIGN EVIDENCE"}
            </span>
            <h1>
              {mechanicsTest ? "RoboArm Mechanics Replay" : "RoboArm Attempt Replay"}
            </h1>
          </div>
        </div>
        <div className="topbar-status">
          <span className="profile-chip">ROARM-M2-S · AUTHORITATIVE PYTHON</span>
          <span className="seed-chip">SEED {String(attempt.seed).padStart(3, "0")}</span>
          <span className={`live-chip ${running ? "is-live" : ""}`}>
            <i />
            {snapshot.success
              ? mechanicsTest
                ? "MECHANICS PASS"
                : "PROMOTION VERIFIED"
              : failed
                ? mechanicsTest
                  ? "EXPECTED REJECTION"
                  : "FAILED ATTEMPT"
                : running
                  ? "REPLAYING"
                  : "EVIDENCE READY"}
          </span>
        </div>
      </header>

      <section className="run-strip" aria-label="Campaign replay controls">
        <div className="run-copy">
          <span className="eyebrow">
            {mechanicsTest
              ? "SCRIPTED REGRESSION · NOT LEARNING"
              : attempt.disposition === "failed"
                ? "GENUINE UNSUCCESSFUL ATTEMPT"
                : "INDEPENDENT FRESH REPLAY"}
          </span>
          <strong>{phaseLabel(attempt, cursor, snapshot)}</strong>
        </div>
        <div className="run-progress">
          <div className="progress-meta">
            <span>
              ACTION {String(cursor).padStart(2, "0")} / {attempt.actions.length}
            </span>
            <span>
              ROLE {moment.role.toUpperCase()} · REWARD {moment.levelsCompleted}
            </span>
          </div>
          <div className="progress-track">
            <span style={{ width: `${progress}%` }} />
          </div>
        </div>
        <div className="run-actions">
          <button
            type="button"
            className="primary-run"
            onClick={startOrResume}
            disabled={running}
            data-testid="start-run"
          >
            <span className="play-symbol">▶</span>
            {atEnd ? "Replay attempt" : cursor > 0 ? "Resume replay" : "Play evidence"}
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => setRunning(false)}
            disabled={!running}
            title="Pause replay"
            data-testid="pause-run"
          >
            ‖
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => seek(cursor + 1)}
            disabled={running || atEnd}
            title="Step recorded action"
            data-testid="step-run"
          >
            ▷|
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => seek(0)}
            title="Return to initial observation"
            data-testid="reset-run"
          >
            ↺
          </button>
        </div>
      </section>

      <div className="workspace-grid">
        <section className="camera-panel">
          <div className="camera-header">
            <div>
              <span className="camera-index">CAM 01</span>
              <span className="camera-title">
                STATE-SYNCHRONIZED 3D REPLAY · HUMAN VIEW
              </span>
            </div>
            <div className="camera-signal">
              <span>40° LENS</span>
              <span>ARTIFACT-LOCKED</span>
              <span className="rec"><i /> REPLAY</span>
            </div>
          </div>
          <div className="camera-viewport">
            <RobotScene
              key={`${selectedIndex}-${attempt.seed}`}
              snapshot={snapshot}
            />
            <div className="sensor-vignette" />
            <div className="sensor-scanlines" />
            <div className="reticle" aria-hidden="true">
              <span className="reticle-h" />
              <span className="reticle-v" />
              <i />
            </div>
            <div className="camera-overlay top-left">
              <span>{attempt.game_id.toUpperCase()} / {attempt.trace_role}</span>
              <strong>{snapshot.sceneId.toUpperCase()}</strong>
            </div>
            <div className="camera-overlay top-right">
              <span>FRAME SHA</span>
              <strong>{shortHash(moment.frameSha256)}</strong>
            </div>
            <div className="camera-overlay bottom-left">
              <span>TCP</span>
              <strong>
                {snapshot.robot.anchors.tcp
                  .map((value) => (value * 1000).toFixed(0))
                  .join(" / ")} MM
              </strong>
            </div>
            <div className="camera-overlay bottom-right">
              <span>OBJECT</span>
              <strong>
                {snapshot.object.attached
                  ? "ATTACHED"
                  : snapshot.object.settled
                    ? "SUPPORTED"
                    : "FREE"}
              </strong>
            </div>
            {failed && (
              <div className="failure-overlay" data-testid="failure-overlay">
                <span className="failure-mark">×</span>
                <div>
                  <small>OBSERVED OUTCOME</small>
                  <strong>
                    {mechanicsTest ? "COLLISION REJECTED" : "HYPOTHESIS FAILED"}
                  </strong>
                  <span>
                    {mechanicsTest
                      ? "Expected mechanics-test outcome · no learning claim"
                      : "No sparse reward · evidence retained"}
                  </span>
                </div>
              </div>
            )}
            {snapshot.success && (
              <div className="success-overlay" data-testid="success-overlay">
                <span className="success-check">✓</span>
                <div>
                  <small>
                    {mechanicsTest ? "PYTHON MECHANICS REPLAY" : "FRESH REPLAY GATE"}
                  </small>
                  <strong>
                    {mechanicsTest ? "MECHANICS FIXTURE PASSED" : "PROMOTION VERIFIED"}
                  </strong>
                  <span>
                    Released · gravity settled · target supported
                    {mechanicsTest ? " · test only" : ""}
                  </span>
                </div>
              </div>
            )}
          </div>
          <div className="camera-footer">
            <span><i className="online" /> PYTHON SNAPSHOT ONLINE</span>
            <span>Drag to orbit · display interpolation only</span>
            <span>NO PLANNING · NO REPAIR · NO SOLVER</span>
          </div>
        </section>

        <aside className="right-rail">
          <RgbObservation
            frame={moment.frame}
            turn={snapshot.turn}
            frameHash={moment.frameSha256}
            mechanicsTest={mechanicsTest}
          />
          <EventTrace snapshot={snapshot} />
        </aside>
      </div>

      <div className="bottom-grid evidence-bottom-grid">
        <Telemetry telemetry={moment.telemetry} />
        <EvidencePanel
          bundle={bundle}
          selectedIndex={selectedIndex}
          attempt={attempt}
          cursor={cursor}
          onSelect={selectAttempt}
          onSeek={seek}
          mechanicsTest={mechanicsTest}
        />
      </div>

      {!mechanicsTest && bundle.lineage !== undefined && (
        <LineagePanel profile={bundle.lineage} />
      )}

      <footer className="footer-note">
        <span>
          {mechanicsTest ? "FIXTURE SET" : "CAMPAIGN"}{" "}
          <strong>{bundle.manifest.campaign_id}</strong>
        </span>
        <span>
          EXACT ACTIONS · RGB CAMERA + TELEMETRY · REPLAY-CHECKED SNAPSHOTS
        </span>
        <span>
          ARC-STYLE CONTRACT <strong>STANDALONE / NO ARC API</strong>
        </span>
      </footer>
    </main>
  );
}
