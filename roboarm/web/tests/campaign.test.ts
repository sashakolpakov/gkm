import { describe, expect, it } from "vitest";

import {
  decodeRgbFrame,
  parseCampaignAttempt,
  parseCampaignLineageProfile,
  parseCampaignManifest,
  replayMoments,
} from "../lib/campaign";
import type { PublicTelemetry, WorldSnapshot } from "../lib/model";

function encodedFrame(value: number): string {
  return Buffer.alloc(72 * 128 * 3, value).toString("base64");
}

const sensorMetadata = {
  sensor_contract_id: "rb01-roarm-c920-v3",
  frame_encoding: "rgb8",
  frame_shape: [72, 128, 3],
  camera_model: { device: "Logitech C920s Pro HD" },
} as const;

function telemetry(turn: number): PublicTelemetry {
  return {
    schema_version: 3,
    sensor_contract_id: "rb01-roarm-c920-v3",
    mode: "operational",
    sample: {
      sequence: turn,
      host_time_s: turn * 0.25 + 0.008,
      arm_request_time_s: turn * 0.25 + 0.002,
      arm_response_time_s: turn * 0.25 + 0.008,
      camera_capture_time_s: turn * 0.25,
      sensor_skew_ms: 8,
    },
    controller: {
      selected_coordinate: "azimuth",
      last_action: turn === 0 ? 0 : 4,
      last_step_duration_s: turn === 0 ? 0 : 0.25,
      command_json: { T: 104, x: 300, y: 0, z: 270, t: 1.08, spd: 0.25 },
      interlocked: false,
    },
    arm: {
      device: "Waveshare RoArm-M2-S",
      transport: "USB serial JSONL",
      baud: 115200,
      request: { T: 105 },
      feedback: {
        T: 1051, x: 300, y: 0, z: 270, b: 0, s: 0, e: 1.57, t: 1.08,
        torB: 8, torS: 95, torE: 48, torH: 4,
        torswitchB: 1, torswitchS: 1, torswitchE: 1, torswitchH: 1, v: 1199,
      },
    },
    camera: {
      device: "Logitech C920s Pro HD",
      transport: "USB UVC",
      source_format: "MJPG",
      source_shape: [1080, 1920, 3],
      source_fps: 30,
      sequence: turn * 8,
      capture_time_s: turn * 0.25,
      observation_encoding: "rgb8",
      observation_shape: [72, 128, 3],
      autofocus: true,
      auto_light_correction: true,
      audio_in_observation: false,
    },
  };
}

function snapshot(turn: number, eventKind?: string): WorldSnapshot {
  const events =
    eventKind === undefined
      ? []
      : [{ turn, kind: eventKind, detail: `event-${turn}` }];
  return {
    schemaVersion: 1,
    sceneId: "pick-place-v1",
    seed: 0,
    turn,
    simulationTimeS: turn * 0.25,
    robot: {
      joints: [0, 0, 0],
      command: { azimuth: 0, reach: 0.3, height: 0.27 },
      selectedAxis: "azimuth",
      gripperOpen: true,
      gripperAperture: 0.08,
      contactLoad: 0,
      rejected: false,
      rejectionReason: "",
      anchors: {
        base: [0, 0, 0],
        shoulder: [0, 0, 0.1],
        elbow: [0.1, 0, 0.2],
        wrist: [0.2, 0, 0.2],
        tcp: [0.3, 0, 0.27],
      },
    },
    object: {
      id: "workpiece",
      position: [0.3, 0, 0.025],
      size: [0.04, 0.04, 0.05],
      massKg: 0.08,
      attached: false,
      settled: true,
    },
    barrier: {
      id: "barrier",
      center: [0.28, 0.1, 0.0625],
      size: [0.075, 0.09, 0.125],
    },
    target: {
      id: "target",
      center: [0.23, 0.19, 0.009],
      size: [0.12, 0.12, 0.018],
    },
    lastAction: turn === 0 ? 0 : 4,
    events,
    eventLog: [],
    success: false,
    terminal: false,
  };
}

function attemptValue() {
  return {
    schema_version: 3,
    attempt_kind: "gkm",
    disposition: "failed",
    trace_role: "committed",
    observed_failure_evidence: ["empty_grasp"],
    replay_stage: "failed_preflight",
    game_id: "rb01-v1",
    scenario: "round-1",
    seed: 0,
    ...sensorMetadata,
    initial_frame_sha256: "a".repeat(64),
    initial_frame_b64: encodedFrame(0),
    initial_telemetry: telemetry(0),
    initial_telemetry_sha256: "d".repeat(64),
    initial_visual_state: snapshot(0),
    actions: [4, 4],
    steps: [
      {
        action: 4,
        frame_b64: encodedFrame(1),
        frame_sha256: "b".repeat(64),
        telemetry: telemetry(1),
        telemetry_sha256: "e".repeat(64),
        levels_completed: 0,
        terminal: false,
        role: "committed",
        visual_state: snapshot(1, "axis_selected"),
      },
      {
        action: 4,
        frame_b64: encodedFrame(2),
        frame_sha256: "c".repeat(64),
        telemetry: telemetry(2),
        telemetry_sha256: "f".repeat(64),
        levels_completed: 0,
        terminal: false,
        role: "committed",
        visual_state: snapshot(2, "axis_selected"),
      },
    ],
  };
}

function lineageValue() {
  const generation = (number: number, overrides = {}) => ({
    generation: number,
    winning_checkpoint: number === 2,
    historical_net_growth: number === 1 ? 200 : 90,
    positive_meaningful_line_additions: 80,
    conditional_ast_zlib_bytes: number === 1 ? 3000 : 1400,
    previous_conditional_ast_zlib_bytes: number === 1 ? null : 3000,
    conditional_ast_drop_bytes: number === 1 ? null : 1600,
    conditional_ast_ratio: number === 1 ? null : 1400 / 3000,
    sharp_marginal_drop: number === 2,
    literal_reused_top_level_nodes: number * 3,
    novel_top_level_nodes: 4,
    direct_unchanged_called_legs: [],
    transitive_unchanged_called_legs:
      number === 1 ? [] : ["legs.py:retained_leg"],
    hard_direct_reuse_witness: false,
    transitive_reuse_witness: number === 2,
    sharp_drop_with_direct_reuse: false,
    milestone: number === 2 ? "verified win" : "admitted revision",
    ...overrides,
  });
  return {
    schema_version: 1,
    profile_kind: "campaign-construction-lineage",
    campaign_id: "test",
    source_boundary: "clean-admitted proposer generation",
    metric_contract: {
      historical_net_growth: "positive net growth",
      conditional_ast_zlib_bytes: "compressed novel AST units",
    },
    interpretation: {
      solved_level_sawtooth_claim: false,
      construction_profile_only: true,
      reason: "one promoted round; generations are construction states",
      historical_net_growth_direction_changes: 0,
      conditional_ast_direction_changes: 0,
      direct_reuse_generations: 0,
      transitive_reuse_generations: 1,
      sharp_direct_coupled_witnesses: 0,
    },
    generations: [generation(1), generation(2)],
    profile_receipt_sha256: "a".repeat(64),
  };
}

describe("campaign replay evidence", () => {
  it("decodes the exact authoritative RGB camera bytes", () => {
    const attempt = parseCampaignAttempt(attemptValue());
    const moment = replayMoments(attempt)[0];
    expect(moment.frame).toBeInstanceOf(Uint8Array);
    expect(moment.frame).toHaveLength(72 * 128 * 3);
    expect(moment.telemetry.sensor_contract_id).toBe(
      "rb01-roarm-c920-v3",
    );
  });

  it("rejects the obsolete indexed campaign schema and frame width", () => {
    expect(() =>
      parseCampaignManifest({
        schema_version: 1,
        export_kind: "replay-validated-gkm-evidence",
        campaign_id: "obsolete",
        attempts: ["failed_attempt.json", "successful_attempt.json"],
        sensor_contract_id: "rb01-indexed-v1",
        frame_encoding: "palette8",
        frame_shape: [64, 64],
        camera_model: {},
      }),
    ).toThrow(/unsupported campaign manifest schema/);
    expect(() =>
      decodeRgbFrame(Buffer.alloc(64 * 64, 0).toString("base64")),
    ).toThrow(/128×72×3 RGB8/);
  });

  it("rejects private fields smuggled into controller telemetry", () => {
    const value = attemptValue();
    value.initial_telemetry = {
      ...value.initial_telemetry,
      object_position: [0.3, 0, 0.025],
    } as PublicTelemetry;
    expect(() => parseCampaignAttempt(value)).toThrow(
      /fields violate the public controller contract/,
    );
  });

  it("requires both failed and promoted attempt entries", () => {
    expect(() =>
      parseCampaignManifest({
        schema_version: 3,
        export_kind: "replay-validated-gkm-evidence",
        campaign_id: "test",
        attempts: ["successful_attempt.json"],
        ...sensorMetadata,
      }),
    ).toThrow(/failure and success/);
    expect(
      parseCampaignManifest({
        schema_version: 3,
        export_kind: "replay-validated-gkm-evidence",
        campaign_id: "test",
        attempts: ["failed_attempt.json", "successful_attempt.json"],
        ...sensorMetadata,
        lineage_profile: "lineage_profile.json",
        lineage_profile_receipt_sha256: "a".repeat(64),
      }).lineage_profile,
    ).toBe("lineage_profile.json");
    expect(
      parseCampaignManifest({
        schema_version: 3,
        export_kind: "replay-validated-gkm-evidence",
        campaign_id: "test",
        attempts: ["failed_attempt.json", "successful_attempt.json"],
        ...sensorMetadata,
      }).attempts,
    ).toHaveLength(2);
  });

  it("validates construction lineage without promoting it to solved-level evidence", () => {
    const lineage = parseCampaignLineageProfile(lineageValue());
    expect(lineage.generations.map((row) => row.historical_net_growth)).toEqual([
      200, 90,
    ]);
    expect(lineage.generations[1].transitive_unchanged_called_legs).toEqual([
      "legs.py:retained_leg",
    ]);
    expect(lineage.interpretation.solved_level_sawtooth_claim).toBe(false);

    expect(() =>
      parseCampaignLineageProfile({
        ...lineageValue(),
        interpretation: {
          ...lineageValue().interpretation,
          solved_level_sawtooth_claim: true,
        },
      }),
    ).toThrow(/overstates solved-level evidence/);
    expect(() =>
      parseCampaignLineageProfile({
        ...lineageValue(),
        generations: [
          lineageValue().generations[1],
          lineageValue().generations[0],
        ],
      }),
    ).toThrow(/positive and increasing/);
  });

  it("keeps mechanics fixtures out of machine acquisition evidence", () => {
    const manifest = {
      schema_version: 3,
      export_kind: "developer-mechanics-test",
      campaign_id: "canonical-mechanics-fixture",
      attempts: ["collision_attempt.json", "successful_replay.json"],
      ...sensorMetadata,
    };
    expect(() => parseCampaignManifest(manifest)).toThrow(
      /replay-validated-gkm-evidence/,
    );
    expect(
      parseCampaignManifest(manifest, "mechanics-test").export_kind,
    ).toBe("developer-mechanics-test");

    const value = {
      ...attemptValue(),
      attempt_kind: "mechanics-test",
      disposition: "expected-rejection",
      fixture_id: "low-clearance-barrier-collision",
    };
    expect(() => parseCampaignAttempt(value)).toThrow(/unsupported/);
    expect(
      parseCampaignAttempt(value, "mechanics-test").fixture_id,
    ).toBe("low-clearance-barrier-collision");
  });

  it("keeps exact frames and accumulates authoritative mechanics events", () => {
    const attempt = parseCampaignAttempt(attemptValue());
    const moments = replayMoments(attempt);
    expect(moments).toHaveLength(3);
    expect(moments[0].frame).toHaveLength(27648);
    expect(moments[2].frame[0]).toBe(2);
    expect(moments[2].telemetry.sample.sequence).toBe(2);
    expect(moments[2].snapshot.eventLog.map((event) => event.turn)).toEqual([
      1, 2,
    ]);
  });

  it("accepts null for optional evidence fields emitted by early failures", () => {
    const value = {
      ...attemptValue(),
      campaign_id: "campaign",
      attempt_id: "g001-s001-probe",
      scenario_id: "g1.probe",
      generation: null,
      source_tree_sha256: null,
      promotion_receipt_sha256: null,
      replay_receipt_sha256: null,
    };
    const attempt = parseCampaignAttempt(value);
    expect(attempt.campaign_id).toBe("campaign");
    expect(attempt.generation).toBeUndefined();
    expect(attempt.source_tree_sha256).toBeUndefined();
    expect(attempt.promotion_receipt_sha256).toBeUndefined();
    expect(attempt.replay_receipt_sha256).toBeUndefined();
  });

  it("rejects action evidence that is not aligned with recorded steps", () => {
    const value = attemptValue();
    value.actions = [3, 4];
    expect(() => parseCampaignAttempt(value)).toThrow(/not aligned/);
  });
});
