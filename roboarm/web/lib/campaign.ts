import type {
  Action,
  MechanicsEvent,
  PublicTelemetry,
  WorldSnapshot,
} from "./model";

export const SENSOR_CONTRACT_ID = "rb01-roarm-c920-v3";
export const RGB_FRAME_SHAPE = [72, 128, 3] as const;

export type EvidenceMode = "gkm" | "mechanics-test";
export type AttemptDisposition =
  | "failed"
  | "promoted"
  | "expected-rejection"
  | "completed";

export interface CampaignManifest {
  schema_version: 3;
  export_kind:
    | "replay-validated-gkm-evidence"
    | "developer-mechanics-test";
  campaign_id: string;
  attempts: string[];
  sensor_contract_id: typeof SENSOR_CONTRACT_ID;
  frame_encoding: "rgb8";
  frame_shape: [72, 128, 3];
  camera_model: Record<string, unknown>;
  lineage_profile?: string;
  lineage_profile_receipt_sha256?: string;
}

export interface LineageGeneration {
  generation: number;
  winning_checkpoint: boolean;
  historical_net_growth: number;
  positive_meaningful_line_additions: number;
  conditional_ast_zlib_bytes: number;
  previous_conditional_ast_zlib_bytes?: number;
  conditional_ast_drop_bytes?: number;
  conditional_ast_ratio?: number;
  sharp_marginal_drop: boolean;
  literal_reused_top_level_nodes: number;
  novel_top_level_nodes: number;
  direct_unchanged_called_legs: string[];
  transitive_unchanged_called_legs: string[];
  hard_direct_reuse_witness: boolean;
  transitive_reuse_witness: boolean;
  sharp_drop_with_direct_reuse: boolean;
  milestone: string;
}

export interface CampaignLineageProfile {
  schema_version: 1;
  profile_kind: "campaign-construction-lineage";
  campaign_id: string;
  source_boundary: string;
  metric_contract: Record<string, string>;
  interpretation: {
    solved_level_sawtooth_claim: false;
    construction_profile_only: true;
    reason: string;
    historical_net_growth_direction_changes: number;
    conditional_ast_direction_changes: number;
    direct_reuse_generations: number;
    transitive_reuse_generations: number;
    sharp_direct_coupled_witnesses: number;
  };
  generations: LineageGeneration[];
  profile_receipt_sha256: string;
}

export interface CampaignStep {
  action: Action;
  frame_b64: string;
  frame_sha256: string;
  telemetry: PublicTelemetry;
  telemetry_sha256: string;
  levels_completed: number;
  terminal: boolean;
  role: string;
  visual_state: WorldSnapshot;
}

export interface CampaignAttempt {
  schema_version: 3;
  attempt_kind: EvidenceMode;
  disposition: AttemptDisposition;
  trace_role: string;
  campaign_id?: string;
  fixture_id?: string;
  attempt_id?: string;
  scenario_id?: string;
  generation?: number;
  hypothesis?: string;
  expected_observation?: string;
  observed_failure_evidence: string[];
  replay_stage?: string;
  fsa_receipt_sha256?: string;
  game_id: string;
  scenario: string;
  seed: number;
  source_tree_sha256?: string;
  promotion_receipt_sha256?: string;
  replay_receipt_sha256?: string;
  sensor_contract_id: typeof SENSOR_CONTRACT_ID;
  frame_encoding: "rgb8";
  frame_shape: [72, 128, 3];
  camera_model: Record<string, unknown>;
  initial_frame_sha256: string;
  initial_frame_b64: string;
  initial_telemetry: PublicTelemetry;
  initial_telemetry_sha256: string;
  initial_visual_state: WorldSnapshot;
  actions: Action[];
  steps: CampaignStep[];
}

export interface CampaignBundle {
  manifest: CampaignManifest;
  attempts: CampaignAttempt[];
  lineage?: CampaignLineageProfile;
}

export interface ReplayMoment {
  index: number;
  action: Action | null;
  frame: Uint8Array;
  frameSha256: string;
  telemetry: PublicTelemetry;
  telemetrySha256: string;
  levelsCompleted: number;
  role: string;
  snapshot: WorldSnapshot;
}

function objectValue(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  label: string,
): void {
  const actual = Object.keys(value).sort();
  const required = [...expected].sort();
  if (
    actual.length !== required.length ||
    actual.some((key, index) => key !== required[index])
  ) {
    throw new Error(`${label} fields violate the public controller contract`);
  }
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a nonempty string`);
  }
  return value;
}

function optionalStringValue(
  value: unknown,
  label: string,
): string | undefined {
  return value === undefined || value === null
    ? undefined
    : stringValue(value, label);
}

function integerValue(value: unknown, label: string): number {
  if (!Number.isInteger(value)) {
    throw new Error(`${label} must be an integer`);
  }
  return value as number;
}

function numberValue(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function optionalIntegerValue(
  value: unknown,
  label: string,
): number | undefined {
  return value === undefined || value === null
    ? undefined
    : integerValue(value, label);
}

function nonnegativeIntegerValue(value: unknown, label: string): number {
  const result = integerValue(value, label);
  if (result < 0) {
    throw new Error(`${label} must be nonnegative`);
  }
  return result;
}

function optionalNumberValue(
  value: unknown,
  label: string,
): number | undefined {
  return value === undefined || value === null
    ? undefined
    : numberValue(value, label);
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be boolean`);
  }
  return value;
}

function stringArrayValue(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value.map((item, index) =>
    stringValue(item, `${label}[${index}]`),
  );
}

function sha256Value(value: unknown, label: string): string {
  const result = stringValue(value, label);
  if (!/^[a-f0-9]{64}$/.test(result)) {
    throw new Error(`${label} must be a lowercase SHA-256 digest`);
  }
  return result;
}

function actionValue(value: unknown, label: string): Action {
  const action = integerValue(value, label);
  if (action < 1 || action > 6) {
    throw new Error(`${label} must be one of the six public actions`);
  }
  return action as Action;
}

function vector3Value(value: unknown, label: string): [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${label} must contain three numbers`);
  }
  return value.map((item, index) =>
    numberValue(item, `${label}[${index}]`),
  ) as [number, number, number];
}

function telemetryValue(value: unknown, label: string): PublicTelemetry {
  const packet = objectValue(value, label);
  const sample = objectValue(packet.sample, `${label}.sample`);
  const controller = objectValue(packet.controller, `${label}.controller`);
  const command = objectValue(controller.command_json, `${label}.controller.command_json`);
  const arm = objectValue(packet.arm, `${label}.arm`);
  const request = objectValue(arm.request, `${label}.arm.request`);
  const feedback = objectValue(arm.feedback, `${label}.arm.feedback`);
  const camera = objectValue(packet.camera, `${label}.camera`);
  exactKeys(
    packet,
    ["schema_version", "sensor_contract_id", "mode", "sample", "controller", "arm", "camera"],
    label,
  );
  exactKeys(sample, ["sequence", "host_time_s", "arm_request_time_s", "arm_response_time_s", "camera_capture_time_s", "sensor_skew_ms"], `${label}.sample`);
  exactKeys(controller, ["selected_coordinate", "last_action", "last_step_duration_s", "command_json", "interlocked"], `${label}.controller`);
  exactKeys(command, ["T", "x", "y", "z", "t", "spd"], `${label}.controller.command_json`);
  exactKeys(arm, ["device", "transport", "baud", "request", "feedback"], `${label}.arm`);
  exactKeys(request, ["T"], `${label}.arm.request`);
  exactKeys(feedback, ["T", "x", "y", "z", "b", "s", "e", "t", "torB", "torS", "torE", "torH", "torswitchB", "torswitchS", "torswitchE", "torswitchH", "v"], `${label}.arm.feedback`);
  exactKeys(camera, ["device", "transport", "source_format", "source_shape", "source_fps", "sequence", "capture_time_s", "observation_encoding", "observation_shape", "autofocus", "auto_light_correction", "audio_in_observation"], `${label}.camera`);
  const coordinate = stringValue(
    controller.selected_coordinate,
    `${label}.controller.selected_coordinate`,
  );
  if (
    packet.schema_version !== 3 ||
    packet.sensor_contract_id !== SENSOR_CONTRACT_ID ||
    packet.mode !== "operational" ||
    !["azimuth", "reach", "height"].includes(coordinate) ||
    command.T !== 104 || request.T !== 105 || feedback.T !== 1051 ||
    typeof controller.interlocked !== "boolean" ||
    typeof camera.autofocus !== "boolean" ||
    typeof camera.auto_light_correction !== "boolean" ||
    typeof camera.audio_in_observation !== "boolean"
  ) {
    throw new Error(`${label} violates the public controller contract`);
  }
  const numericObject = (source: Record<string, unknown>, keys: readonly string[], prefix: string) =>
    Object.fromEntries(keys.map((key) => [key, numberValue(source[key], `${prefix}.${key}`)]));
  const sampleNumbers = numericObject(sample, ["sequence", "host_time_s", "arm_request_time_s", "arm_response_time_s", "camera_capture_time_s", "sensor_skew_ms"], `${label}.sample`);
  const commandNumbers = numericObject(command, ["x", "y", "z", "t", "spd"], `${label}.controller.command_json`);
  const feedbackNumbers = numericObject(feedback, ["x", "y", "z", "b", "s", "e", "t", "torB", "torS", "torE", "torH", "torswitchB", "torswitchS", "torswitchE", "torswitchH", "v"], `${label}.arm.feedback`);
  return {
    schema_version: 3,
    sensor_contract_id: SENSOR_CONTRACT_ID,
    mode: "operational",
    sample: sampleNumbers as PublicTelemetry["sample"],
    controller: {
      selected_coordinate: coordinate as PublicTelemetry["controller"]["selected_coordinate"],
      last_action: integerValue(controller.last_action, `${label}.controller.last_action`),
      last_step_duration_s: numberValue(controller.last_step_duration_s, `${label}.controller.last_step_duration_s`),
      command_json: { T: 104, ...commandNumbers } as PublicTelemetry["controller"]["command_json"],
      interlocked: controller.interlocked,
    },
    arm: {
      device: stringValue(arm.device, `${label}.arm.device`),
      transport: stringValue(arm.transport, `${label}.arm.transport`),
      baud: integerValue(arm.baud, `${label}.arm.baud`),
      request: { T: 105 },
      feedback: { T: 1051, ...feedbackNumbers } as PublicTelemetry["arm"]["feedback"],
    },
    camera: {
      device: stringValue(camera.device, `${label}.camera.device`),
      transport: stringValue(camera.transport, `${label}.camera.transport`),
      source_format: stringValue(camera.source_format, `${label}.camera.source_format`),
      source_shape: vector3Value(camera.source_shape, `${label}.camera.source_shape`),
      source_fps: numberValue(camera.source_fps, `${label}.camera.source_fps`),
      sequence: integerValue(camera.sequence, `${label}.camera.sequence`),
      capture_time_s: numberValue(camera.capture_time_s, `${label}.camera.capture_time_s`),
      observation_encoding: "rgb8",
      observation_shape: vector3Value(camera.observation_shape, `${label}.camera.observation_shape`),
      autofocus: camera.autofocus,
      auto_light_correction: camera.auto_light_correction,
      audio_in_observation: camera.audio_in_observation,
    },
  };
}

function sensorMetadata(
  data: Record<string, unknown>,
  label: string,
): {
  sensor_contract_id: typeof SENSOR_CONTRACT_ID;
  frame_encoding: "rgb8";
  frame_shape: [72, 128, 3];
  camera_model: Record<string, unknown>;
} {
  if (
    data.sensor_contract_id !== SENSOR_CONTRACT_ID ||
    data.frame_encoding !== "rgb8" ||
    !Array.isArray(data.frame_shape) ||
    data.frame_shape.length !== 3 ||
    data.frame_shape.some(
      (value, index) => value !== RGB_FRAME_SHAPE[index],
    )
  ) {
    throw new Error(`${label} uses an obsolete sensor contract`);
  }
  return {
    sensor_contract_id: SENSOR_CONTRACT_ID,
    frame_encoding: "rgb8",
    frame_shape: [72, 128, 3],
    camera_model: objectValue(data.camera_model, `${label}.camera_model`),
  };
}

function snapshotValue(value: unknown, label: string): WorldSnapshot {
  const snapshot = objectValue(value, label);
  const robot = objectValue(snapshot.robot, `${label}.robot`);
  const object = objectValue(snapshot.object, `${label}.object`);
  const barrier = objectValue(snapshot.barrier, `${label}.barrier`);
  const target = objectValue(snapshot.target, `${label}.target`);
  const events = snapshot.events;
  if (
    !Array.isArray(robot.joints) ||
    !Array.isArray(object.position) ||
    !Array.isArray(barrier.center) ||
    !Array.isArray(target.center) ||
    !Array.isArray(events)
  ) {
    throw new Error(`${label} is missing authoritative visual-state fields`);
  }
  return {
    ...(snapshot as unknown as WorldSnapshot),
    eventLog: [],
  };
}

export function parseCampaignManifest(
  value: unknown,
  evidenceMode: EvidenceMode = "gkm",
): CampaignManifest {
  const data = objectValue(value, "campaign manifest");
  if (data.schema_version !== 3 || !Array.isArray(data.attempts)) {
    throw new Error("unsupported campaign manifest schema");
  }
  const sensors = sensorMetadata(data, "campaign manifest");
  const expectedExportKind =
    evidenceMode === "gkm"
      ? "replay-validated-gkm-evidence"
      : "developer-mechanics-test";
  if (data.export_kind !== expectedExportKind) {
    throw new Error(
      `campaign manifest is not ${expectedExportKind}`,
    );
  }
  const attempts = data.attempts.map((item, index) =>
    stringValue(item, `manifest.attempts[${index}]`),
  );
  if (attempts.length < 2) {
    throw new Error("campaign manifest must include failure and success evidence");
  }
  for (const filename of attempts) {
    if (!/^[a-z0-9_-]+\.json$/i.test(filename)) {
      throw new Error("campaign attempt filename is unsafe");
    }
  }
  const lineageProfile = optionalStringValue(
    data.lineage_profile,
    "manifest.lineage_profile",
  );
  const lineageReceipt = data.lineage_profile_receipt_sha256 === undefined
    ? undefined
    : sha256Value(
        data.lineage_profile_receipt_sha256,
        "manifest.lineage_profile_receipt_sha256",
      );
  if ((lineageProfile === undefined) !== (lineageReceipt === undefined)) {
    throw new Error("campaign lineage filename and receipt must appear together");
  }
  if (
    lineageProfile !== undefined &&
    (!/^[a-z0-9_-]+\.json$/i.test(lineageProfile) || evidenceMode !== "gkm")
  ) {
    throw new Error("campaign lineage filename is unsafe or out of scope");
  }
  return {
    schema_version: 3,
    export_kind: expectedExportKind,
    campaign_id: stringValue(data.campaign_id, "manifest.campaign_id"),
    attempts,
    ...sensors,
    lineage_profile: lineageProfile,
    lineage_profile_receipt_sha256: lineageReceipt,
  };
}

export function parseCampaignLineageProfile(
  value: unknown,
): CampaignLineageProfile {
  const data = objectValue(value, "campaign lineage profile");
  if (
    data.schema_version !== 1 ||
    data.profile_kind !== "campaign-construction-lineage" ||
    !Array.isArray(data.generations) ||
    data.generations.length === 0
  ) {
    throw new Error("unsupported campaign lineage profile schema");
  }
  const rawGenerations = data.generations;
  const interpretation = objectValue(
    data.interpretation,
    "lineage.interpretation",
  );
  if (
    interpretation.solved_level_sawtooth_claim !== false ||
    interpretation.construction_profile_only !== true
  ) {
    throw new Error("lineage profile overstates solved-level evidence");
  }
  const generations = rawGenerations.map((value, index): LineageGeneration => {
    const row = objectValue(value, `lineage.generations[${index}]`);
    const generation = nonnegativeIntegerValue(
      row.generation,
      `lineage.generations[${index}].generation`,
    );
    if (
      generation === 0 ||
      (index > 0 &&
        generation <= integerValue(
          objectValue(rawGenerations[index - 1], "previous lineage generation").generation,
          "previous lineage generation number",
        ))
    ) {
      throw new Error("lineage generations must be positive and increasing");
    }
    const direct = stringArrayValue(
      row.direct_unchanged_called_legs,
      `lineage.generations[${index}].direct_unchanged_called_legs`,
    );
    const transitive = stringArrayValue(
      row.transitive_unchanged_called_legs,
      `lineage.generations[${index}].transitive_unchanged_called_legs`,
    );
    return {
      generation,
      winning_checkpoint: booleanValue(row.winning_checkpoint, "lineage winning checkpoint"),
      historical_net_growth: nonnegativeIntegerValue(row.historical_net_growth, "lineage historical net growth"),
      positive_meaningful_line_additions: nonnegativeIntegerValue(row.positive_meaningful_line_additions, "lineage positive line additions"),
      conditional_ast_zlib_bytes: nonnegativeIntegerValue(row.conditional_ast_zlib_bytes, "lineage conditional AST bytes"),
      previous_conditional_ast_zlib_bytes: optionalNumberValue(row.previous_conditional_ast_zlib_bytes, "lineage previous conditional AST bytes"),
      conditional_ast_drop_bytes: optionalNumberValue(row.conditional_ast_drop_bytes, "lineage conditional AST drop"),
      conditional_ast_ratio: optionalNumberValue(row.conditional_ast_ratio, "lineage conditional AST ratio"),
      sharp_marginal_drop: booleanValue(row.sharp_marginal_drop, "lineage sharp marginal drop"),
      literal_reused_top_level_nodes: nonnegativeIntegerValue(row.literal_reused_top_level_nodes, "lineage reused AST units"),
      novel_top_level_nodes: nonnegativeIntegerValue(row.novel_top_level_nodes, "lineage novel AST units"),
      direct_unchanged_called_legs: direct,
      transitive_unchanged_called_legs: transitive,
      hard_direct_reuse_witness: booleanValue(row.hard_direct_reuse_witness, "lineage direct reuse witness"),
      transitive_reuse_witness: booleanValue(row.transitive_reuse_witness, "lineage transitive reuse witness"),
      sharp_drop_with_direct_reuse: booleanValue(row.sharp_drop_with_direct_reuse, "lineage coupled witness"),
      milestone: stringValue(row.milestone, "lineage milestone"),
    };
  });
  const metricContract = objectValue(data.metric_contract, "lineage.metric_contract");
  const metrics = Object.fromEntries(
    Object.entries(metricContract).map(([key, metric]) => [
      key,
      stringValue(metric, `lineage.metric_contract.${key}`),
    ]),
  );
  return {
    schema_version: 1,
    profile_kind: "campaign-construction-lineage",
    campaign_id: stringValue(data.campaign_id, "lineage.campaign_id"),
    source_boundary: stringValue(data.source_boundary, "lineage.source_boundary"),
    metric_contract: metrics,
    interpretation: {
      solved_level_sawtooth_claim: false,
      construction_profile_only: true,
      reason: stringValue(interpretation.reason, "lineage.interpretation.reason"),
      historical_net_growth_direction_changes: nonnegativeIntegerValue(interpretation.historical_net_growth_direction_changes, "lineage net-growth direction changes"),
      conditional_ast_direction_changes: nonnegativeIntegerValue(interpretation.conditional_ast_direction_changes, "lineage AST direction changes"),
      direct_reuse_generations: nonnegativeIntegerValue(interpretation.direct_reuse_generations, "lineage direct reuse generations"),
      transitive_reuse_generations: nonnegativeIntegerValue(interpretation.transitive_reuse_generations, "lineage transitive reuse generations"),
      sharp_direct_coupled_witnesses: nonnegativeIntegerValue(interpretation.sharp_direct_coupled_witnesses, "lineage coupled witnesses"),
    },
    generations,
    profile_receipt_sha256: sha256Value(
      data.profile_receipt_sha256,
      "lineage.profile_receipt_sha256",
    ),
  };
}

export function parseCampaignAttempt(
  value: unknown,
  evidenceMode: EvidenceMode = "gkm",
): CampaignAttempt {
  const data = objectValue(value, "campaign attempt");
  const validDisposition =
    evidenceMode === "gkm"
      ? data.disposition === "failed" || data.disposition === "promoted"
      : data.disposition === "expected-rejection" ||
        data.disposition === "completed";
  if (
    data.schema_version !== 3 ||
    data.attempt_kind !== evidenceMode ||
    !validDisposition ||
    !Array.isArray(data.actions) ||
    !Array.isArray(data.steps)
  ) {
    throw new Error("unsupported campaign attempt schema");
  }
  const sensors = sensorMetadata(data, "campaign attempt");
  const actions = data.actions.map((item, index) =>
    actionValue(item, `attempt.actions[${index}]`),
  );
  const steps = data.steps.map((item, index): CampaignStep => {
    const step = objectValue(item, `attempt.steps[${index}]`);
    return {
      action: actionValue(step.action, `attempt.steps[${index}].action`),
      frame_b64: stringValue(
        step.frame_b64,
        `attempt.steps[${index}].frame_b64`,
      ),
      frame_sha256: stringValue(
        step.frame_sha256,
        `attempt.steps[${index}].frame_sha256`,
      ),
      telemetry: telemetryValue(
        step.telemetry,
        `attempt.steps[${index}].telemetry`,
      ),
      telemetry_sha256: stringValue(
        step.telemetry_sha256,
        `attempt.steps[${index}].telemetry_sha256`,
      ),
      levels_completed: integerValue(
        step.levels_completed,
        `attempt.steps[${index}].levels_completed`,
      ),
      terminal: Boolean(step.terminal),
      role: stringValue(step.role, `attempt.steps[${index}].role`),
      visual_state: snapshotValue(
        step.visual_state,
        `attempt.steps[${index}].visual_state`,
      ),
    };
  });
  if (
    actions.length === 0 ||
    actions.length !== steps.length ||
    steps.some((step, index) => step.action !== actions[index])
  ) {
    throw new Error("campaign action and step evidence are not aligned");
  }
  const rawFailureEvidence = data.observed_failure_evidence ?? [];
  if (
    !Array.isArray(rawFailureEvidence) ||
    rawFailureEvidence.some(
      (item) => typeof item !== "string" || item.length === 0,
    )
  ) {
    throw new Error("campaign failure evidence is invalid");
  }
  const observedFailureEvidence = rawFailureEvidence as string[];
  if (evidenceMode === "gkm") {
    if (data.disposition === "failed" && observedFailureEvidence.length === 0) {
      throw new Error("failed campaign attempt has no operational failure");
    }
    if (data.disposition === "promoted" && observedFailureEvidence.length > 0) {
      throw new Error("promoted campaign attempt contains failure evidence");
    }
  }
  const generation = optionalIntegerValue(
    data.generation,
    "attempt.generation",
  );
  if (generation !== undefined && generation <= 0) {
    throw new Error("attempt.generation must be positive");
  }
  return {
    schema_version: 3,
    attempt_kind: evidenceMode,
    disposition: data.disposition as AttemptDisposition,
    trace_role: stringValue(data.trace_role, "attempt.trace_role"),
    campaign_id: optionalStringValue(
      data.campaign_id,
      "attempt.campaign_id",
    ),
    fixture_id: optionalStringValue(data.fixture_id, "attempt.fixture_id"),
    attempt_id: optionalStringValue(data.attempt_id, "attempt.attempt_id"),
    scenario_id: optionalStringValue(data.scenario_id, "attempt.scenario_id"),
    generation,
    hypothesis: optionalStringValue(
      data.hypothesis,
      "attempt.hypothesis",
    ),
    expected_observation: optionalStringValue(
      data.expected_observation,
      "attempt.expected_observation",
    ),
    observed_failure_evidence: observedFailureEvidence,
    replay_stage: optionalStringValue(
      data.replay_stage,
      "attempt.replay_stage",
    ),
    fsa_receipt_sha256: optionalStringValue(
      data.fsa_receipt_sha256,
      "attempt.fsa_receipt_sha256",
    ),
    game_id: stringValue(data.game_id, "attempt.game_id"),
    scenario: stringValue(data.scenario, "attempt.scenario"),
    seed: integerValue(data.seed, "attempt.seed"),
    source_tree_sha256: optionalStringValue(
      data.source_tree_sha256,
      "attempt.source_tree_sha256",
    ),
    promotion_receipt_sha256: optionalStringValue(
      data.promotion_receipt_sha256,
      "attempt.promotion_receipt_sha256",
    ),
    replay_receipt_sha256: optionalStringValue(
      data.replay_receipt_sha256,
      "attempt.replay_receipt_sha256",
    ),
    ...sensors,
    initial_frame_sha256: stringValue(
      data.initial_frame_sha256,
      "attempt.initial_frame_sha256",
    ),
    initial_frame_b64: stringValue(
      data.initial_frame_b64,
      "attempt.initial_frame_b64",
    ),
    initial_telemetry: telemetryValue(
      data.initial_telemetry,
      "attempt.initial_telemetry",
    ),
    initial_telemetry_sha256: stringValue(
      data.initial_telemetry_sha256,
      "attempt.initial_telemetry_sha256",
    ),
    initial_visual_state: snapshotValue(
      data.initial_visual_state,
      "attempt.initial_visual_state",
    ),
    actions,
    steps,
  };
}

export function decodeRgbFrame(encoded: string): Uint8Array {
  const binary = atob(encoded);
  if (binary.length !== 72 * 128 * 3) {
    throw new Error("campaign frame is not exactly 128×72×3 RGB8 bytes");
  }
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function withEventLog(
  snapshot: WorldSnapshot,
  eventLog: MechanicsEvent[],
): WorldSnapshot {
  const events = snapshot.events.map((event) => ({ ...event }));
  return {
    ...snapshot,
    events,
    eventLog: [...eventLog, ...events],
  };
}

export function replayMoments(attempt: CampaignAttempt): ReplayMoment[] {
  const initial = withEventLog(attempt.initial_visual_state, []);
  const moments: ReplayMoment[] = [
    {
      index: 0,
      action: null,
      frame: decodeRgbFrame(attempt.initial_frame_b64),
      frameSha256: attempt.initial_frame_sha256,
      telemetry: attempt.initial_telemetry,
      telemetrySha256: attempt.initial_telemetry_sha256,
      levelsCompleted: 0,
      role: attempt.trace_role,
      snapshot: initial,
    },
  ];
  let eventLog: MechanicsEvent[] = [];
  for (const [index, step] of attempt.steps.entries()) {
    const snapshot = withEventLog(step.visual_state, eventLog);
    eventLog = snapshot.eventLog;
    moments.push({
      index: index + 1,
      action: step.action,
      frame: decodeRgbFrame(step.frame_b64),
      frameSha256: step.frame_sha256,
      telemetry: step.telemetry,
      telemetrySha256: step.telemetry_sha256,
      levelsCompleted: step.levels_completed,
      role: step.role,
      snapshot,
    });
  }
  return moments;
}

export async function loadCampaignBundle(
  basePath = "/campaign",
  evidenceMode: EvidenceMode = "gkm",
): Promise<CampaignBundle> {
  const manifestResponse = await fetch(`${basePath}/manifest.json`, {
    cache: "no-store",
  });
  if (!manifestResponse.ok) {
    throw new Error(
      "no admitted Godel-Kolmogorov machine campaign has been exported",
    );
  }
  const manifest = parseCampaignManifest(
    await manifestResponse.json(),
    evidenceMode,
  );
  const attempts = await Promise.all(
    manifest.attempts.map(async (filename) => {
      const response = await fetch(`${basePath}/${filename}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`campaign attempt ${filename} is unavailable`);
      }
      const attempt = parseCampaignAttempt(
        await response.json(),
        evidenceMode,
      );
      if (attempt.campaign_id !== manifest.campaign_id) {
        throw new Error(`campaign attempt ${filename} has the wrong campaign id`);
      }
      return attempt;
    }),
  );
  let lineage: CampaignLineageProfile | undefined;
  if (manifest.lineage_profile !== undefined) {
    const response = await fetch(`${basePath}/${manifest.lineage_profile}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error("campaign lineage profile is unavailable");
    }
    lineage = parseCampaignLineageProfile(await response.json());
    if (
      lineage.campaign_id !== manifest.campaign_id ||
      lineage.profile_receipt_sha256 !==
        manifest.lineage_profile_receipt_sha256
    ) {
      throw new Error("campaign lineage profile does not match its manifest");
    }
  }
  if (evidenceMode === "gkm") {
    if (
      !attempts.some((attempt) => attempt.disposition === "failed") ||
      !attempts.some((attempt) => attempt.disposition === "promoted")
    ) {
      throw new Error(
        "viewer evidence must contain a genuine failure and promotion",
      );
    }
  } else if (
    !attempts.some(
      (attempt) => attempt.disposition === "expected-rejection",
    ) ||
    !attempts.some((attempt) => attempt.disposition === "completed")
  ) {
    throw new Error(
      "mechanics evidence must contain a rejection and completed replay",
    );
  }
  return { manifest, attempts, lineage };
}
