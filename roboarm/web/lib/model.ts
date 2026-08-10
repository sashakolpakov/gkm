export type Vec3 = [number, number, number];
export type JointVector = [number, number, number];
export type Coordinate = "azimuth" | "reach" | "height";
export type Action = 1 | 2 | 3 | 4 | 5 | 6;

export interface ArmAnchors {
  base: Vec3;
  shoulder: Vec3;
  elbow: Vec3;
  wrist: Vec3;
  tcp: Vec3;
}

export interface MechanicsEvent {
  turn: number;
  kind: string;
  detail: string;
}

export interface PublicTelemetry {
  schema_version: 3;
  sensor_contract_id: "rb01-roarm-c920-v3";
  mode: "operational";
  sample: {
    sequence: number;
    host_time_s: number;
    arm_request_time_s: number;
    arm_response_time_s: number;
    camera_capture_time_s: number;
    sensor_skew_ms: number;
  };
  controller: {
    selected_coordinate: Coordinate;
    last_action: number;
    last_step_duration_s: number;
    command_json: {
      T: 104;
      x: number;
      y: number;
      z: number;
      t: number;
      spd: number;
    };
    interlocked: boolean;
  };
  arm: {
    device: string;
    transport: string;
    baud: number;
    request: { T: 105 };
    feedback: {
      T: 1051;
      x: number;
      y: number;
      z: number;
      b: number;
      s: number;
      e: number;
      t: number;
      torB: number;
      torS: number;
      torE: number;
      torH: number;
      torswitchB: number;
      torswitchS: number;
      torswitchE: number;
      torswitchH: number;
      v: number;
    };
  };
  camera: {
    device: string;
    transport: string;
    source_format: string;
    source_shape: [number, number, number];
    source_fps: number;
    sequence: number;
    capture_time_s: number;
    observation_encoding: "rgb8";
    observation_shape: [number, number, number];
    autofocus: boolean;
    auto_light_correction: boolean;
    audio_in_observation: boolean;
  };
}

export interface SceneBox {
  id: string;
  center: Vec3;
  size: Vec3;
}

export interface RobotState {
  joints: JointVector;
  command: {
    azimuth: number;
    reach: number;
    height: number;
  };
  selectedAxis: Coordinate;
  gripperOpen: boolean;
  gripperAperture: number;
  contactLoad: number;
  rejected: boolean;
  rejectionReason: string;
}

export interface ObjectState {
  id: string;
  position: Vec3;
  size: Vec3;
  massKg: number;
  attached: boolean;
  settled: boolean;
}

export interface WorldState {
  robot: RobotState;
  object: ObjectState;
  barrier: SceneBox;
  target: SceneBox;
  attachedOffset: Vec3 | null;
  actionCount: number;
  simulationTimeS: number;
  success: boolean;
  levelFailed: boolean;
  lastAction: number;
  lastEvents: MechanicsEvent[];
  eventLog: MechanicsEvent[];
}

export interface WorldSnapshot {
  schemaVersion: number;
  sceneId: string;
  seed: number;
  turn: number;
  simulationTimeS: number;
  robot: RobotState & { anchors: ArmAnchors };
  object: ObjectState;
  barrier: SceneBox;
  target: SceneBox;
  lastAction: number;
  events: MechanicsEvent[];
  eventLog: MechanicsEvent[];
  success: boolean;
  terminal: boolean;
}

export const ACTIONS: readonly Action[] = [1, 2, 3, 4, 5, 6] as const;
export const COORDINATES: readonly Coordinate[] = [
  "azimuth",
  "reach",
  "height",
] as const;

export const ACTION_LABELS: Record<Action, string> = {
  1: "Decrease",
  2: "Increase",
  3: "Previous axis",
  4: "Next axis",
  5: "Open gripper",
  6: "Close gripper",
};

export const SCENE_SCHEMA_VERSION = 2;
export const CANONICAL_SCENE_ID = "pick-place-v2";

export const BASE_AXIS_X_OFFSET_M = 0.0100000008759151;
export const SHOULDER_HEIGHT_M = 0.123059270461044;
export const UPPER_ARM_X_M = 0.236815132922094;
export const UPPER_ARM_RADIAL_OFFSET_M = 0.0300023995170449;
export const UPPER_ARM_EFFECTIVE_M = Math.hypot(
  UPPER_ARM_X_M,
  UPPER_ARM_RADIAL_OFFSET_M,
);
export const FOREARM_TCP_X_M = 0.002;
export const FOREARM_TCP_Z_M = 0.2802;
export const FOREARM_TO_TCP_EFFECTIVE_M = Math.hypot(
  FOREARM_TCP_Z_M,
  FOREARM_TCP_X_M,
);

export const Q0_LIMITS: readonly [number, number] = [-3.1416, 3.1416];
export const Q1_LIMITS: readonly [number, number] = [-1.5708, 1.5708];
export const Q2_LIMITS: readonly [number, number] = [-1.0, 2.95];

export const COMMAND_AZIMUTH_LIMITS_RAD: readonly [number, number] = [
  (-80 * Math.PI) / 180,
  (80 * Math.PI) / 180,
];
export const COMMAND_REACH_LIMITS_M: readonly [number, number] = [0.16, 0.48];
export const COMMAND_HEIGHT_LIMITS_M: readonly [number, number] = [0.045, 0.52];

export const INITIAL_AZIMUTH_RAD = 0;
export const INITIAL_REACH_M = 0.3;
export const INITIAL_HEIGHT_M = 0.27;

export const TABLE_Z_M = 0;
export const TABLE_SIZE_M: Vec3 = [0.9, 0.9, 0.055];
export const BASE_RADIUS_M = 0.065;
export const BASE_HEIGHT_M = 0.035;
export const BASE_COLUMN_RADIUS_M = 0.026;
export const UPPER_ARM_RADIUS_M = 0.026;
export const FOREARM_RADIUS_M = 0.021;
export const WRIST_LINK_RADIUS_M = 0.017;
export const SHOULDER_JOINT_RADIUS_M = 0.035;
export const ELBOW_JOINT_RADIUS_M = 0.031;
export const WRIST_JOINT_RADIUS_M = 0.024;
export const ROBOT_LINK_RADIUS_M = UPPER_ARM_RADIUS_M;
export const GRIPPER_RADIUS_M = 0.018;
export const GRIPPER_OPEN_APERTURE_M = 0.08;
export const GRIPPER_CLOSED_APERTURE_M = 0.008;
export const GRIPPER_JAW_DEPTH_M = 0.065;
export const GRIPPER_VERTICAL_TOLERANCE_M = 0.03;
export const GRIPPER_PALM_SIZE_M: Vec3 = [0.06, 0.085, 0.024];
export const GRIPPER_PALM_RADIAL_OFFSET_M = -0.026;
export const GRIPPER_JAW_SIZE_M: Vec3 = [0.06, 0.009, 0.04];
export const GRIPPER_JAW_RADIAL_OFFSET_M = 0.004;
export const GRIPPER_JAW_VERTICAL_OFFSET_M = -0.02;
export const SWEEP_MAX_JOINT_DELTA_RAD = 0.025;

export const OBJECT_SIZE_M: Vec3 = [0.04, 0.04, 0.05];
export const OBJECT_MASS_KG = 0.08;
export const OBJECT_START_AZIMUTH_RAD = 0;
export const OBJECT_START_REACH_M = 0.3;

export const TARGET_AZIMUTH_RAD = (60 * Math.PI) / 180;
export const TARGET_REACH_M = 0.3;
export const TARGET_SIZE_M: Vec3 = [0.14, 0.14, 0.018];
export const TARGET_WALL_HEIGHT_M = 0.048;
export const TARGET_WALL_THICKNESS_M = 0.006;

export const BARRIER_AZIMUTH_RAD = (20 * Math.PI) / 180;
export const BARRIER_REACH_M = 0.3;
export const BARRIER_SIZE_M: Vec3 = [0.075, 0.09, 0.125];
export const BARRIER_CAP_OVERHANG_M = 0.003;
export const BARRIER_CAP_THICKNESS_M = 0.009;

export const WORKCELL_REAR_WALL_CENTER_M: Vec3 = [0, 0.63, 0.315];
export const WORKCELL_REAR_WALL_SIZE_M: Vec3 = [1.55, 0.035, 0.85];
export const WORKCELL_POST_X_M: readonly [number, number] = [-0.48, 0.48];
export const WORKCELL_POST_Y_M = 0.48;
export const WORKCELL_POST_SIZE_M: Vec3 = [0.018, 0.018, 0.6];

export const AZIMUTH_STEP_RAD = (5 * Math.PI) / 180;
export const REACH_STEP_M = 0.02;
export const HEIGHT_STEP_M = 0.015;
export const CONTROL_TURN_S = 0.25;
export const KINEMATIC_TOLERANCE_M = 2e-5;
export const ROARM_ENCODER_STEP_RAD = (2 * Math.PI) / 4096;

export const RATED_PAYLOAD_MOMENT_NM = 0.5 * 9.80665 * 0.5;

export function cloneVec3(value: Vec3): Vec3 {
  return [value[0], value[1], value[2]];
}

export function addVec3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function subtractVec3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
