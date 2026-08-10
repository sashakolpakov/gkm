import { describe, expect, it } from "vitest";

import {
  BASE_AXIS_X_OFFSET_M,
  KINEMATIC_TOLERANCE_M,
  SHOULDER_HEIGHT_M,
} from "../lib/model";
import {
  cylindricalFromTcp,
  exactAnchors,
  interpolateJoints,
  solveCylindrical,
} from "../lib/kinematics";

function expectVectorClose(
  actual: readonly number[],
  expected: readonly number[],
  tolerance: number,
): void {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => {
    expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(tolerance);
  });
}

describe("pinned RoArm-M2-S kinematics", () => {
  it("matches the preserved Xacro zero-pose anchors", () => {
    const anchors = exactAnchors([0, 0, 0]);
    expectVectorClose(
      anchors.shoulder,
      [BASE_AXIS_X_OFFSET_M, 0, SHOULDER_HEIGHT_M],
      1e-12,
    );
    expectVectorClose(
      anchors.elbow,
      [0.040002471103605, -1.102271092340012e-7, 0.359874293621211],
      2e-6,
    );
    expectVectorClose(
      anchors.tcp,
      [0.042001530522195, -1.175548245092273e-7, 0.640074293618902],
      2e-6,
    );
  });

  it("hits the canonical command lattice with the fixed IK branch", () => {
    for (const azimuth of [-40, 0, 40].map((value) => (value * Math.PI) / 180)) {
      for (const reach of [0.24, 0.3, 0.38]) {
        for (const height of [0.045, 0.15, 0.27, 0.36]) {
          const result = solveCylindrical(azimuth, reach, height);
          expect(result).not.toBeNull();
          const observed = cylindricalFromTcp(result!.tcp);
          expect(Math.abs(observed[0] - azimuth)).toBeLessThan(1e-8);
          expect(Math.abs(observed[1] - reach)).toBeLessThan(
            KINEMATIC_TOLERANCE_M,
          );
          expect(Math.abs(observed[2] - height)).toBeLessThan(
            KINEMATIC_TOLERANCE_M,
          );
        }
      }
    }
  });

  it("bounds every swept interpolation and preserves its endpoint", () => {
    const path = interpolateJoints([0, 0, 0], [0.4, -0.3, 0.2], 0.025);
    let previous: [number, number, number] = [0, 0, 0];
    for (const current of path) {
      expect(
        Math.max(...current.map((value, index) => Math.abs(value - previous[index]))),
      ).toBeLessThanOrEqual(0.025 + 1e-12);
      previous = current;
    }
    expectVectorClose(path.at(-1)!, [0.4, -0.3, 0.2], 1e-15);
  });
});
