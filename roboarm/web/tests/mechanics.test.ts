import { describe, expect, it } from "vitest";

import parity from "../../references/operational_parity.json";
import { CANONICAL_PICK_PLACE_ACTIONS } from "../lib/canonical";
import { OperationalWorld } from "../lib/mechanics";

function expectVectorClose(
  actual: readonly number[],
  expected: readonly number[],
  tolerance = 2e-11,
): void {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => {
    expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(tolerance);
  });
}

describe("operational world parity", () => {
  it("matches Python mechanics and events at camera-hashed key turns", () => {
    const world = new OperationalWorld(0);
    const keyframes = new Map(parity.keyframes.map((entry) => [entry.turn, entry]));

    const check = (): void => {
      const expected = keyframes.get(world.state.actionCount);
      if (expected === undefined) return;
      const snapshot = world.snapshot();
      expectVectorClose(snapshot.robot.joints, expected.joints);
      expectVectorClose(snapshot.robot.anchors.tcp, expected.tcp);
      expectVectorClose(snapshot.object.position, expected.object);
      expect(snapshot.robot.gripperAperture).toBeCloseTo(expected.aperture, 14);
      expect(snapshot.object.attached).toBe(expected.attached);
      expect(snapshot.success).toBe(expected.success);
      expect(snapshot.events).toEqual(expected.events);
      expect(expected.frameSha256).toMatch(/^[a-f0-9]{64}$/);
    };

    check();
    for (const action of CANONICAL_PICK_PLACE_ACTIONS) {
      world.step(action);
      check();
    }
    expect(world.state.actionCount).toBe(parity.actionCount);
  });

  it("completes grasp, carry, release, settlement, and target verification", () => {
    const world = new OperationalWorld(0);
    for (const action of CANONICAL_PICK_PLACE_ACTIONS) {
      world.step(action);
    }
    const snapshot = world.snapshot();
    const kinds = snapshot.eventLog.map((event) => event.kind);
    expect(snapshot.turn).toBe(63);
    expect(snapshot.success).toBe(true);
    expect(snapshot.terminal).toBe(true);
    expect(snapshot.object.attached).toBe(false);
    expect(snapshot.object.settled).toBe(true);
    const objectBottom =
      snapshot.object.position[2] - snapshot.object.size[2] * 0.5;
    const targetTop =
      snapshot.target.center[2] + snapshot.target.size[2] * 0.5;
    expect(objectBottom).toBeCloseTo(targetTop, 14);
    expect(kinds).toContain("jaw_contact_left");
    expect(kinds).toContain("jaw_contact_right");
    expect(kinds).toContain("object_attached");
    expect(kinds.slice(-3)).toEqual([
      "object_released",
      "gravity_settled",
      "level_completed",
    ]);
  });

  it("rejects a carried descent through the physical target floor", () => {
    const world = new OperationalWorld(0);
    for (const action of CANONICAL_PICK_PLACE_ACTIONS.slice(0, -3)) {
      world.step(action);
    }
    const acceptedHeight = world.snapshot().robot.command.height;
    const rejected = world.step(1);
    expect(rejected.robot.rejected).toBe(true);
    expect(rejected.robot.rejectionReason).toBe(
      "held_object_target_collision",
    );
    expect(rejected.robot.command.height).toBe(acceptedHeight);
    expect(rejected.object.attached).toBe(true);
  });

  it("rejects a low full-gripper sweep through the physical barrier", () => {
    const world = new OperationalWorld(0);
    const approachAndGrasp = [
      4,
      4,
      ...Array(15).fill(1),
      6,
      ...Array(7).fill(2),
      4,
    ];
    for (const action of approachAndGrasp) world.step(action);
    expect(world.snapshot().object.attached).toBe(true);

    world.step(2);
    const acceptedAzimuth = world.snapshot().robot.command.azimuth;
    const rejected = world.step(2);
    expect(rejected.robot.rejected).toBe(true);
    expect(rejected.robot.rejectionReason).toBe(
      "gripper_barrier_collision",
    );
    expect(rejected.robot.command.azimuth).toBe(acceptedAzimuth);
  });

  it("keeps every accepted canonical pose free of rendered-solid overlap", () => {
    const world = new OperationalWorld(0);
    expect(world.currentCollisionReason()).toBe("");
    for (const action of CANONICAL_PICK_PLACE_ACTIONS) {
      world.step(action);
      expect(world.currentCollisionReason()).toBe("");
    }
  });

  it("treats the obstacle cap and target walls as authoritative colliders", () => {
    const capWorld = new OperationalWorld(0);
    const capProbe = [2, 4, ...Array(4).fill(1), 4, ...Array(7).fill(1)];
    for (const action of capProbe) capWorld.step(action);
    const acceptedCapHeight = capWorld.snapshot().robot.command.height;
    const rejectedCap = capWorld.step(1);
    expect(rejectedCap.robot.rejectionReason).toBe(
      "gripper_barrier_cap_collision",
    );
    expect(rejectedCap.robot.command.height).toBe(acceptedCapHeight);

    const wallWorld = new OperationalWorld(0);
    const wallProbe = [
      ...Array(8).fill(2),
      4,
      ...Array(4).fill(1),
      4,
      ...Array(13).fill(1),
    ];
    for (const action of wallProbe) wallWorld.step(action);
    const acceptedWallHeight = wallWorld.snapshot().robot.command.height;
    const rejectedWall = wallWorld.step(1);
    expect(rejectedWall.robot.rejectionReason).toBe(
      "gripper_target_wall_collision",
    );
    expect(rejectedWall.robot.command.height).toBe(acceptedWallHeight);
  });

  it("rotates the attachment offset in the gripper frame", () => {
    const world = new OperationalWorld(0);
    const actions = [
      4,
      2,
      4,
      ...Array(15).fill(1),
      6,
      ...Array(14).fill(2),
      4,
      ...Array(4).fill(2),
    ];
    for (const action of actions) world.step(action);
    const snapshot = world.snapshot();
    expect(snapshot.object.attached).toBe(true);
    const yaw = snapshot.robot.command.azimuth;
    const deltaX = snapshot.object.position[0] - snapshot.robot.anchors.tcp[0];
    const deltaY = snapshot.object.position[1] - snapshot.robot.anchors.tcp[1];
    const radialOffset = deltaX * Math.cos(yaw) + deltaY * Math.sin(yaw);
    const tangentialOffset =
      -deltaX * Math.sin(yaw) + deltaY * Math.cos(yaw);
    expect(Math.abs(radialOffset + 0.02)).toBeLessThanOrEqual(4e-5);
    expect(Math.abs(tangentialOffset)).toBeLessThanOrEqual(4e-5);
  });

  it("deep-clones state and diverges without aliasing", () => {
    const original = new OperationalWorld(0);
    for (const action of CANONICAL_PICK_PLACE_ACTIONS.slice(0, 18)) {
      original.step(action);
    }
    const clone = original.clone();
    expect(clone.snapshot()).toEqual(original.snapshot());
    clone.step(2);
    expect(clone.snapshot()).not.toEqual(original.snapshot());
    expect(original.snapshot().turn).toBe(18);
  });
});
