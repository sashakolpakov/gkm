import {
  BASE_AXIS_X_OFFSET_M,
  FOREARM_TCP_X_M,
  FOREARM_TCP_Z_M,
  FOREARM_TO_TCP_EFFECTIVE_M,
  KINEMATIC_TOLERANCE_M,
  Q0_LIMITS,
  Q1_LIMITS,
  Q2_LIMITS,
  SHOULDER_HEIGHT_M,
  UPPER_ARM_EFFECTIVE_M,
  UPPER_ARM_RADIAL_OFFSET_M,
  UPPER_ARM_X_M,
  type ArmAnchors,
  type JointVector,
  type Vec3,
} from "./model";

export type Matrix4 = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

export interface IKResult {
  joints: JointVector;
  tcp: Vec3;
  errorM: number;
}

export const IDENTITY: Matrix4 = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

export function multiply(left: Matrix4, right: Matrix4): Matrix4 {
  const result = new Array<number>(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      for (let inner = 0; inner < 4; inner += 1) {
        result[row * 4 + column] +=
          left[row * 4 + inner] * right[inner * 4 + column];
      }
    }
  }
  return result as Matrix4;
}

export function rotationX(angle: number): Matrix4 {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    1, 0, 0, 0,
    0, cosine, -sine, 0,
    0, sine, cosine, 0,
    0, 0, 0, 1,
  ];
}

export function rotationY(angle: number): Matrix4 {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    cosine, 0, sine, 0,
    0, 1, 0, 0,
    -sine, 0, cosine, 0,
    0, 0, 0, 1,
  ];
}

export function rotationZ(angle: number): Matrix4 {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    cosine, -sine, 0, 0,
    sine, cosine, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
}

export function translation(x: number, y: number, z: number): Matrix4 {
  return [
    1, 0, 0, x,
    0, 1, 0, y,
    0, 0, 1, z,
    0, 0, 0, 1,
  ];
}

export function rpy(roll: number, pitch: number, yaw: number): Matrix4 {
  return multiply(multiply(rotationZ(yaw), rotationY(pitch)), rotationX(roll));
}

function point(transform: Matrix4): Vec3 {
  return [transform[3], transform[7], transform[11]];
}

export function exactTransforms(joints: JointVector): Record<string, Matrix4> {
  const [q0, q1, q2] = joints;
  const world = [...IDENTITY] as Matrix4;
  const base = world;
  const link1 = multiply(
    multiply(
      base,
      translation(BASE_AXIS_X_OFFSET_M, 0, SHOULDER_HEIGHT_M),
    ),
    rotationZ(q0),
  );
  const link2 = multiply(
    multiply(link1, rpy(-1.5708, -1.5708, 0)),
    rotationZ(q1),
  );
  const link3 = multiply(
    multiply(
      multiply(
        link2,
        translation(UPPER_ARM_X_M, UPPER_ARM_RADIAL_OFFSET_M, 0),
      ),
      rpy(0, 0, 1.5708),
    ),
    rotationZ(q2),
  );
  const gripper = multiply(
    multiply(
      link3,
      translation(0.002906, -0.21599, -0.00066683),
    ),
    rpy(-1.5708, 0, -1.5708),
  );
  const tcp = multiply(
    multiply(
      link3,
      translation(FOREARM_TCP_X_M, -FOREARM_TCP_Z_M, 0),
    ),
    rpy(1.5708, 0, -1.5708),
  );
  return {
    world,
    base_link: base,
    link1,
    link2,
    link3,
    gripper_link: gripper,
    hand_tcp: tcp,
  };
}

export function exactAnchors(joints: JointVector): ArmAnchors {
  const transforms = exactTransforms(joints);
  return {
    base: [0, 0, 0],
    shoulder: point(transforms.link1),
    elbow: point(transforms.link3),
    wrist: point(transforms.gripper_link),
    tcp: point(transforms.hand_tcp),
  };
}

export function cylindricalFromTcp(tcp: Vec3): [number, number, number] {
  const relativeX = tcp[0] - BASE_AXIS_X_OFFSET_M;
  return [Math.atan2(tcp[1], relativeX), Math.hypot(relativeX, tcp[1]), tcp[2]];
}

const UPPER_OFFSET = Math.atan2(
  UPPER_ARM_RADIAL_OFFSET_M,
  UPPER_ARM_X_M,
);
const FOREARM_OFFSET = Math.atan2(FOREARM_TCP_X_M, FOREARM_TCP_Z_M);
const EXACT_AZIMUTH_BIAS = cylindricalFromTcp(exactAnchors([0, 0, 0]).tcp)[0];

function within(value: number, limits: readonly [number, number]): boolean {
  return limits[0] - 1e-9 <= value && value <= limits[1] + 1e-9;
}

function analyticSeed(reach: number, height: number): [number, number] | null {
  const vertical = height - SHOULDER_HEIGHT_M;
  const numerator =
    reach * reach +
    vertical * vertical -
    UPPER_ARM_EFFECTIVE_M * UPPER_ARM_EFFECTIVE_M -
    FOREARM_TO_TCP_EFFECTIVE_M * FOREARM_TO_TCP_EFFECTIVE_M;
  const denominator =
    2 * UPPER_ARM_EFFECTIVE_M * FOREARM_TO_TCP_EFFECTIVE_M;
  let cosineDelta = numerator / denominator;
  if (cosineDelta < -1 - 1e-12 || cosineDelta > 1 + 1e-12) {
    return null;
  }
  cosineDelta = Math.min(1, Math.max(-1, cosineDelta));
  const delta = Math.acos(cosineDelta);
  const theta1 =
    Math.atan2(reach, vertical) -
    Math.atan2(
      FOREARM_TO_TCP_EFFECTIVE_M * Math.sin(delta),
      UPPER_ARM_EFFECTIVE_M +
        FOREARM_TO_TCP_EFFECTIVE_M * Math.cos(delta),
    );
  const q1 = theta1 - UPPER_OFFSET;
  const q2 = delta + UPPER_OFFSET - FOREARM_OFFSET;
  return [q1, q2];
}

export function solveCylindrical(
  azimuth: number,
  reach: number,
  height: number,
): IKResult | null {
  const seed = analyticSeed(reach, height);
  if (seed === null) {
    return null;
  }
  let [q1, q2] = seed;
  let q0 = azimuth - EXACT_AZIMUTH_BIAS;

  for (let iteration = 0; iteration < 8; iteration += 1) {
    const tcp = exactAnchors([q0, q1, q2]).tcp;
    const [observedAzimuth, observedReach, observedHeight] =
      cylindricalFromTcp(tcp);
    const residualReach = reach - observedReach;
    const residualHeight = height - observedHeight;
    if (Math.hypot(residualReach, residualHeight) <= KINEMATIC_TOLERANCE_M * 0.1) {
      q0 += azimuth - observedAzimuth;
      break;
    }

    const epsilon = 1e-6;
    const shiftedQ1 = cylindricalFromTcp(
      exactAnchors([q0, q1 + epsilon, q2]).tcp,
    );
    const shiftedQ2 = cylindricalFromTcp(
      exactAnchors([q0, q1, q2 + epsilon]).tcp,
    );
    const a00 = (shiftedQ1[1] - observedReach) / epsilon;
    const a10 = (shiftedQ1[2] - observedHeight) / epsilon;
    const a01 = (shiftedQ2[1] - observedReach) / epsilon;
    const a11 = (shiftedQ2[2] - observedHeight) / epsilon;
    const determinant = a00 * a11 - a01 * a10;
    if (Math.abs(determinant) < 1e-15) {
      return null;
    }
    q1 += (residualReach * a11 - a01 * residualHeight) / determinant;
    q2 += (a00 * residualHeight - residualReach * a10) / determinant;
    q0 += azimuth - observedAzimuth;
  }

  const joints: JointVector = [q0, q1, q2];
  if (
    !within(q0, Q0_LIMITS) ||
    !within(q1, Q1_LIMITS) ||
    !within(q2, Q2_LIMITS)
  ) {
    return null;
  }
  const tcp = exactAnchors(joints).tcp;
  const [observedAzimuth, observedReach, observedHeight] =
    cylindricalFromTcp(tcp);
  const angularError = Math.abs(
    Math.atan2(
      Math.sin(observedAzimuth - azimuth),
      Math.cos(observedAzimuth - azimuth),
    ),
  );
  const errorM = Math.hypot(
    Math.hypot(observedReach - reach, observedHeight - height),
    angularError * Math.max(reach, 1e-6),
  );
  if (errorM > KINEMATIC_TOLERANCE_M) {
    return null;
  }
  return { joints, tcp, errorM };
}

export function interpolateJoints(
  start: JointVector,
  end: JointVector,
  maxDelta: number,
): JointVector[] {
  const largest = Math.max(
    ...end.map((value, index) => Math.abs(value - start[index])),
  );
  const steps = Math.max(1, Math.ceil(largest / maxDelta));
  return Array.from({ length: steps }, (_, zeroIndex) => {
    const ratio = (zeroIndex + 1) / steps;
    return start.map(
      (value, index) => value + (end[index] - value) * ratio,
    ) as JointVector;
  });
}
