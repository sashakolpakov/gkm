import {
  ACTIONS,
  AZIMUTH_STEP_RAD,
  BARRIER_AZIMUTH_RAD,
  BARRIER_REACH_M,
  BARRIER_SIZE_M,
  BASE_AXIS_X_OFFSET_M,
  BASE_HEIGHT_M,
  BASE_RADIUS_M,
  CANONICAL_SCENE_ID,
  COMMAND_AZIMUTH_LIMITS_RAD,
  COMMAND_HEIGHT_LIMITS_M,
  COMMAND_REACH_LIMITS_M,
  CONTROL_TURN_S,
  COORDINATES,
  ELBOW_JOINT_RADIUS_M,
  GRIPPER_CLOSED_APERTURE_M,
  GRIPPER_JAW_DEPTH_M,
  GRIPPER_OPEN_APERTURE_M,
  GRIPPER_VERTICAL_TOLERANCE_M,
  HEIGHT_STEP_M,
  INITIAL_AZIMUTH_RAD,
  INITIAL_HEIGHT_M,
  INITIAL_REACH_M,
  OBJECT_MASS_KG,
  OBJECT_SIZE_M,
  OBJECT_START_AZIMUTH_RAD,
  OBJECT_START_REACH_M,
  RATED_PAYLOAD_MOMENT_NM,
  REACH_STEP_M,
  ROARM_ENCODER_STEP_RAD,
  SCENE_SCHEMA_VERSION,
  SWEEP_MAX_JOINT_DELTA_RAD,
  TABLE_Z_M,
  TARGET_AZIMUTH_RAD,
  TARGET_REACH_M,
  TARGET_SIZE_M,
  WRIST_JOINT_RADIUS_M,
  cloneVec3,
  type Action,
  type ArmAnchors,
  type MechanicsEvent,
  type SceneBox,
  type Vec3,
  type WorldSnapshot,
  type WorldState,
} from "./model";
import {
  boxesOverlap,
  capsulesOverlap,
  insideHorizontalTarget,
  pointSegmentDistance,
  segmentIntersectsBox,
  segmentIntersectsVerticalCylinder,
  segmentIntersectsYawBox,
  sphereIntersectsBox,
  sphereIntersectsVerticalCylinder,
  yawBoxIntersectsBox,
  yawBoxIntersectsVerticalCylinder,
} from "./geometry";
import {
  cylindricalFromTcp,
  exactAnchors,
  interpolateJoints,
  solveCylindrical,
} from "./kinematics";
import {
  attachedWorldPosition,
  attachmentLocalOffset,
  barrierCap,
  gripperBoxes,
  robotCapsules,
  targetWalls,
  workcellSolids,
} from "./physicalGeometry";

export function polarPoint(azimuth: number, reach: number, z: number): Vec3 {
  return [
    BASE_AXIS_X_OFFSET_M + reach * Math.cos(azimuth),
    reach * Math.sin(azimuth),
    z,
  ];
}

function quantizeJoints(joints: Vec3): Vec3 {
  return joints.map(
    (value) => Math.round(value / ROARM_ENCODER_STEP_RAD) * ROARM_ENCODER_STEP_RAD,
  ) as Vec3;
}

function sceneBox(id: string, center: Vec3, size: Vec3): SceneBox {
  return { id, center: cloneVec3(center), size: cloneVec3(size) };
}

export function canonicalState(): WorldState {
  const solution = solveCylindrical(
    INITIAL_AZIMUTH_RAD,
    INITIAL_REACH_M,
    INITIAL_HEIGHT_M,
  );
  if (solution === null) {
    throw new Error("canonical initial command is unreachable");
  }
  return {
    robot: {
      joints: quantizeJoints(solution.joints),
      command: {
        azimuth: INITIAL_AZIMUTH_RAD,
        reach: INITIAL_REACH_M,
        height: INITIAL_HEIGHT_M,
      },
      selectedAxis: "azimuth",
      gripperOpen: true,
      gripperAperture: GRIPPER_OPEN_APERTURE_M,
      contactLoad: 0,
      rejected: false,
      rejectionReason: "",
    },
    object: {
      id: "workpiece",
      position: polarPoint(
        OBJECT_START_AZIMUTH_RAD,
        OBJECT_START_REACH_M,
        TABLE_Z_M + OBJECT_SIZE_M[2] * 0.5,
      ),
      size: cloneVec3(OBJECT_SIZE_M),
      massKg: OBJECT_MASS_KG,
      attached: false,
      settled: true,
    },
    barrier: sceneBox(
      "barrier",
      polarPoint(
        BARRIER_AZIMUTH_RAD,
        BARRIER_REACH_M,
        TABLE_Z_M + BARRIER_SIZE_M[2] * 0.5,
      ),
      BARRIER_SIZE_M,
    ),
    target: sceneBox(
      "target-bin",
      polarPoint(
        TARGET_AZIMUTH_RAD,
        TARGET_REACH_M,
        TABLE_Z_M + TARGET_SIZE_M[2] * 0.5,
      ),
      TARGET_SIZE_M,
    ),
    attachedOffset: null,
    actionCount: 0,
    simulationTimeS: 0,
    success: false,
    levelFailed: false,
    lastAction: 0,
    lastEvents: [],
    eventLog: [],
  };
}

function cloneEvent(event: MechanicsEvent): MechanicsEvent {
  return { turn: event.turn, kind: event.kind, detail: event.detail };
}

export function cloneWorldState(state: WorldState): WorldState {
  return {
    robot: {
      ...state.robot,
      joints: [...state.robot.joints],
      command: { ...state.robot.command },
    },
    object: {
      ...state.object,
      position: cloneVec3(state.object.position),
      size: cloneVec3(state.object.size),
    },
    barrier: sceneBox(state.barrier.id, state.barrier.center, state.barrier.size),
    target: sceneBox(state.target.id, state.target.center, state.target.size),
    attachedOffset:
      state.attachedOffset === null ? null : cloneVec3(state.attachedOffset),
    actionCount: state.actionCount,
    simulationTimeS: state.simulationTimeS,
    success: state.success,
    levelFailed: state.levelFailed,
    lastAction: state.lastAction,
    lastEvents: state.lastEvents.map(cloneEvent),
    eventLog: state.eventLog.map(cloneEvent),
  };
}

function isAction(value: number): value is Action {
  return ACTIONS.includes(value as Action);
}

function round9(value: number): number {
  return Math.round(value * 1e9) / 1e9;
}

export class OperationalWorld {
  readonly seed: number;
  state: WorldState;

  constructor(seed = 0) {
    if (!Number.isInteger(seed)) {
      throw new TypeError("seed must be an integer");
    }
    this.seed = seed;
    this.state = canonicalState();
  }

  reset(): WorldSnapshot {
    this.state = canonicalState();
    return this.snapshot();
  }

  clone(): OperationalWorld {
    const copy = new OperationalWorld(this.seed);
    copy.state = cloneWorldState(this.state);
    return copy;
  }

  currentCollisionReason(): string {
    return this.configurationCollision(
      exactAnchors(this.state.robot.joints),
      this.state.robot.gripperAperture,
    );
  }

  private event(kind: string, detail: string): MechanicsEvent {
    return { turn: this.state.actionCount, kind, detail };
  }

  private finishTurn(action: Action, events: MechanicsEvent[]): void {
    this.state.lastAction = action;
    this.state.simulationTimeS = this.state.actionCount * CONTROL_TURN_S;
    this.state.lastEvents = events.map(cloneEvent);
    this.state.eventLog.push(...events.map(cloneEvent));
  }

  step(action: number): WorldSnapshot {
    if (!Number.isInteger(action) || !isAction(action)) {
      throw new RangeError(`invalid action ${action}; expected one of ${ACTIONS}`);
    }
    if (this.state.success || this.state.levelFailed) {
      throw new Error("cannot step a terminal operational scene");
    }

    this.state.actionCount += 1;
    this.state.robot.rejected = false;
    this.state.robot.rejectionReason = "";
    this.state.robot.contactLoad = 0;
    let events: MechanicsEvent[];

    if (action === 3 || action === 4) {
      const selectedIndex = COORDINATES.indexOf(this.state.robot.selectedAxis);
      const direction = action === 3 ? -1 : 1;
      const nextIndex =
        (selectedIndex + direction + COORDINATES.length) % COORDINATES.length;
      this.state.robot.selectedAxis = COORDINATES[nextIndex];
      events = [
        this.event("axis_selected", this.state.robot.selectedAxis),
      ];
    } else if (action === 5) {
      events = this.openGripper();
    } else if (action === 6) {
      events = this.closeGripper();
    } else {
      events = this.coordinateMotion(action === 1 ? -1 : 1);
    }

    this.finishTurn(action, events);
    return this.snapshot();
  }

  private coordinateMotion(direction: -1 | 1): MechanicsEvent[] {
    const robot = this.state.robot;
    let { azimuth, reach, height } = robot.command;
    if (robot.selectedAxis === "azimuth") {
      azimuth += direction * AZIMUTH_STEP_RAD;
    } else if (robot.selectedAxis === "reach") {
      reach += direction * REACH_STEP_M;
    } else {
      height += direction * HEIGHT_STEP_M;
    }

    if (
      azimuth < COMMAND_AZIMUTH_LIMITS_RAD[0] ||
      azimuth > COMMAND_AZIMUTH_LIMITS_RAD[1] ||
      reach < COMMAND_REACH_LIMITS_M[0] ||
      reach > COMMAND_REACH_LIMITS_M[1] ||
      height < COMMAND_HEIGHT_LIMITS_M[0] - 1e-9 ||
      height > COMMAND_HEIGHT_LIMITS_M[1]
    ) {
      return [this.reject("command_bounds")];
    }

    const solution = solveCylindrical(azimuth, reach, height);
    if (solution === null) {
      return [this.reject("inverse_kinematics")];
    }
    const candidateJoints = quantizeJoints(solution.joints);
    const legality = this.sweptLegality(candidateJoints);
    if (!legality.legal) {
      return [this.reject(legality.reason)];
    }

    const priorTcp = exactAnchors(robot.joints).tcp;
    robot.joints = candidateJoints;
    robot.command = { azimuth, reach, height };
    if (this.state.object.attached) {
      if (this.state.attachedOffset === null) {
        throw new Error("attached object is missing its relative offset");
      }
      const finalYaw = cylindricalFromTcp(legality.finalAnchors.tcp)[0];
      this.state.object.position = attachedWorldPosition(
        legality.finalAnchors.tcp,
        finalYaw,
        this.state.attachedOffset,
      );
      this.state.object.settled = false;
      robot.contactLoad = this.normalizedLoad();
    }
    const displacement = Math.hypot(
      legality.finalAnchors.tcp[0] - priorTcp[0],
      legality.finalAnchors.tcp[1] - priorTcp[1],
      legality.finalAnchors.tcp[2] - priorTcp[2],
    );
    return [
      this.event(
        "motion_accepted",
        `${robot.selectedAxis} displacement=${displacement.toFixed(6)}m`,
      ),
    ];
  }

  private reject(reason: string): MechanicsEvent {
    this.state.robot.rejected = true;
    this.state.robot.rejectionReason = reason;
    this.state.robot.contactLoad = 1;
    return this.event("motion_rejected", reason);
  }

  private sweptLegality(candidateJoints: [number, number, number]): {
    legal: boolean;
    reason: string;
    finalAnchors: ArmAnchors;
  } {
    const finalAnchors = exactAnchors(candidateJoints);
    for (const joints of interpolateJoints(
      this.state.robot.joints,
      candidateJoints,
      SWEEP_MAX_JOINT_DELTA_RAD,
    )) {
      const anchors = exactAnchors(joints);
      const reason = this.configurationCollision(
        anchors,
        this.state.robot.gripperAperture,
      );
      if (reason !== "") {
        return { legal: false, reason, finalAnchors };
      }
    }
    return { legal: true, reason: "", finalAnchors };
  }

  private configurationCollision(
    anchors: ArmAnchors,
    aperture: number,
  ): string {
    const yaw = cylindricalFromTcp(anchors.tcp)[0];
    const [column, upper, forearm, wrist] = robotCapsules(anchors);
    const movingCapsules = [upper, forearm, wrist];
    const movingJoints: Array<[Vec3, number]> = [
      [anchors.elbow, ELBOW_JOINT_RADIUS_M],
      [anchors.wrist, WRIST_JOINT_RADIUS_M],
    ];
    const gripper = gripperBoxes(anchors.tcp, yaw, aperture);

    let objectCenter: Vec3 | null = null;
    let objectBox: SceneBox | null = null;
    if (this.state.object.attached) {
      if (this.state.attachedOffset === null) {
        throw new Error("attached object is missing its relative offset");
      }
      objectCenter = attachedWorldPosition(
        anchors.tcp,
        yaw,
        this.state.attachedOffset,
      );
      objectBox = {
        id: this.state.object.id,
        center: objectCenter,
        size: this.state.object.size,
      };
    }

    if (
      objectCenter !== null &&
      objectCenter[2] - this.state.object.size[2] * 0.5 < TABLE_Z_M - 1e-8
    ) {
      return "held_object_table_collision";
    }
    for (const capsule of movingCapsules) {
      if (
        Math.min(capsule.start[2], capsule.end[2]) - capsule.radius <
        TABLE_Z_M
      ) {
        return "arm_table_collision";
      }
    }
    for (const [center, radius] of movingJoints) {
      if (center[2] - radius < TABLE_Z_M) return "arm_table_collision";
    }
    for (const body of gripper) {
      if (body.center[2] - body.size[2] * 0.5 < TABLE_Z_M) {
        return "gripper_table_collision";
      }
    }

    if (
      objectBox !== null &&
      yawBoxIntersectsVerticalCylinder(
        objectBox.center,
        objectBox.size,
        0,
        0,
        0,
        BASE_RADIUS_M,
        TABLE_Z_M,
        BASE_HEIGHT_M,
      )
    ) {
      return "held_object_base_collision";
    }
    for (const capsule of movingCapsules) {
      if (
        segmentIntersectsVerticalCylinder(
          capsule.start,
          capsule.end,
          capsule.radius,
          0,
          0,
          BASE_RADIUS_M,
          TABLE_Z_M,
          BASE_HEIGHT_M,
        )
      ) {
        return "arm_base_collision";
      }
    }
    for (const [center, radius] of movingJoints) {
      if (
        sphereIntersectsVerticalCylinder(
          center,
          radius,
          0,
          0,
          BASE_RADIUS_M,
          TABLE_Z_M,
          BASE_HEIGHT_M,
        )
      ) {
        return "arm_base_collision";
      }
    }
    for (const body of gripper) {
      if (
        yawBoxIntersectsVerticalCylinder(
          body.center,
          body.size,
          body.yaw,
          0,
          0,
          BASE_RADIUS_M,
          TABLE_Z_M,
          BASE_HEIGHT_M,
        )
      ) {
        return "gripper_base_collision";
      }
    }

    if (
      capsulesOverlap(
        column.start,
        column.end,
        column.radius,
        forearm.start,
        forearm.end,
        forearm.radius,
      ) ||
      capsulesOverlap(
        column.start,
        column.end,
        column.radius,
        wrist.start,
        wrist.end,
        wrist.radius,
      ) ||
      capsulesOverlap(
        upper.start,
        upper.end,
        upper.radius,
        wrist.start,
        wrist.end,
        wrist.radius,
      ) ||
      pointSegmentDistance(anchors.wrist, column.start, column.end) <
        WRIST_JOINT_RADIUS_M + column.radius ||
      pointSegmentDistance(anchors.wrist, upper.start, upper.end) <
        WRIST_JOINT_RADIUS_M + upper.radius
    ) {
      return "arm_self_collision";
    }
    for (const body of gripper) {
      if (
        segmentIntersectsYawBox(
          column.start,
          column.end,
          column.radius,
          body.center,
          body.size,
          body.yaw,
        ) ||
        segmentIntersectsYawBox(
          upper.start,
          upper.end,
          upper.radius,
          body.center,
          body.size,
          body.yaw,
        )
      ) {
        return "gripper_self_collision";
      }
    }
    if (
      objectBox !== null &&
      (segmentIntersectsBox(
        column.start,
        column.end,
        objectBox,
        column.radius,
      ) ||
        segmentIntersectsBox(
          upper.start,
          upper.end,
          objectBox,
          upper.radius,
        ) ||
        segmentIntersectsBox(
          forearm.start,
          forearm.end,
          objectBox,
          forearm.radius,
        ))
    ) {
      return "held_object_self_collision";
    }

    const environment: Array<[string, SceneBox, number]> = [
      ["barrier", this.state.barrier, 0.002],
      ["barrier_cap", barrierCap(this.state.barrier), 0],
      ["target", this.state.target, 0],
      ...targetWalls(this.state.target).map(
        (wall): [string, SceneBox, number] => ["target_wall", wall, 0],
      ),
      ...workcellSolids().map(
        (solid): [string, SceneBox, number] => ["workcell", solid, 0],
      ),
    ];
    for (const [category, solid, objectMargin] of environment) {
      if (
        objectBox !== null &&
        boxesOverlap(
          objectBox.center,
          objectBox.size,
          solid,
          objectMargin,
        )
      ) {
        return `held_object_${category}_collision`;
      }
      for (const capsule of movingCapsules) {
        if (
          segmentIntersectsBox(
            capsule.start,
            capsule.end,
            solid,
            capsule.radius,
          )
        ) {
          return `arm_${category}_collision`;
        }
      }
      for (const [center, radius] of movingJoints) {
        if (sphereIntersectsBox(center, radius, solid)) {
          return `arm_${category}_collision`;
        }
      }
      for (const body of gripper) {
        if (
          yawBoxIntersectsBox(
            body.center,
            body.size,
            body.yaw,
            solid,
          )
        ) {
          return `gripper_${category}_collision`;
        }
      }
    }
    return "";
  }

  private closeGripper(): MechanicsEvent[] {
    const robot = this.state.robot;
    if (this.state.object.attached) {
      robot.gripperOpen = false;
      robot.contactLoad = this.normalizedLoad();
      return [this.event("gripper_closed", "object_already_attached")];
    }

    const tcp = exactAnchors(robot.joints).tcp;
    const object = this.state.object;
    const deltaX = object.position[0] - tcp[0];
    const deltaY = object.position[1] - tcp[1];
    const deltaZ = object.position[2] - tcp[2];
    const radialAxis: [number, number] = [
      Math.cos(robot.command.azimuth),
      Math.sin(robot.command.azimuth),
    ];
    const jawAxis: [number, number] = [-radialAxis[1], radialAxis[0]];
    const depthOffset = Math.abs(
      deltaX * radialAxis[0] + deltaY * radialAxis[1],
    );
    const lateralOffset = Math.abs(
      deltaX * jawAxis[0] + deltaY * jawAxis[1],
    );
    const objectWidth = Math.max(object.size[0], object.size[1]);
    const bilateral =
      depthOffset <= GRIPPER_JAW_DEPTH_M * 0.5 &&
      lateralOffset <= 0.006 &&
      Math.abs(deltaZ) <= GRIPPER_VERTICAL_TOLERANCE_M &&
      objectWidth < GRIPPER_OPEN_APERTURE_M;
    const targetAperture = bilateral
      ? objectWidth
      : GRIPPER_CLOSED_APERTURE_M;
    const collision = this.apertureSweepCollision(
      robot.gripperAperture,
      targetAperture,
    );
    if (collision !== "") {
      return [this.reject(collision)];
    }

    robot.gripperOpen = false;
    if (!bilateral) {
      robot.gripperAperture = targetAperture;
      return [
        this.event("gripper_closed_empty", "bilateral_enclosure_failed"),
      ];
    }

    robot.gripperAperture = targetAperture;
    robot.contactLoad = 0.22;
    object.attached = true;
    object.settled = false;
    this.state.attachedOffset = attachmentLocalOffset(
      tcp,
      robot.command.azimuth,
      object.position,
    );
    return [
      this.event("jaw_contact_left", object.id),
      this.event("jaw_contact_right", object.id),
      this.event("object_attached", object.id),
    ];
  }

  private openGripper(): MechanicsEvent[] {
    const robot = this.state.robot;
    const collision = this.apertureSweepCollision(
      robot.gripperAperture,
      GRIPPER_OPEN_APERTURE_M,
    );
    if (collision !== "") {
      return [this.reject(collision)];
    }
    robot.gripperOpen = true;
    robot.gripperAperture = GRIPPER_OPEN_APERTURE_M;
    if (!this.state.object.attached) {
      return [this.event("gripper_opened", "empty")];
    }

    const object = this.state.object;
    object.attached = false;
    this.state.attachedOffset = null;
    const [supportId, supportTop] = this.highestSupportBelow();
    const settledZ = supportTop + object.size[2] * 0.5;
    object.position = [object.position[0], object.position[1], settledZ];
    object.settled = true;
    const events = [
      this.event("object_released", object.id),
      this.event(
        "gravity_settled",
        `support=${supportId} z=${settledZ.toFixed(6)}`,
      ),
    ];
    if (
      supportId === this.state.target.id &&
      insideHorizontalTarget(object.position, object.size, this.state.target)
    ) {
      this.state.success = true;
      events.push(this.event("level_completed", this.state.target.id));
    }
    return events;
  }

  private apertureSweepCollision(start: number, end: number): string {
    const anchors = exactAnchors(this.state.robot.joints);
    const steps = Math.max(1, Math.ceil(Math.abs(end - start) / 0.004));
    for (let index = 1; index <= steps; index += 1) {
      const aperture = start + ((end - start) * index) / steps;
      const collision = this.configurationCollision(anchors, aperture);
      if (collision !== "") return collision;
    }
    return "";
  }

  private highestSupportBelow(): [string, number] {
    const object = this.state.object;
    const objectBottom = object.position[2] - object.size[2] * 0.5;
    const candidates: [string, number][] = [["table", TABLE_Z_M]];
    for (const support of [this.state.barrier, this.state.target]) {
      const supportTop = support.center[2] + support.size[2] * 0.5;
      if (
        supportTop <= objectBottom + 1e-8 &&
        insideHorizontalTarget(
          object.position,
          object.size,
          support,
          0,
        )
      ) {
        candidates.push([support.id, supportTop]);
      }
    }
    return candidates.reduce((highest, candidate) =>
      candidate[1] > highest[1] ? candidate : highest,
    );
  }

  private normalizedLoad(): number {
    const moment =
      this.state.object.massKg * 9.80665 * this.state.robot.command.reach;
    return Math.min(1, moment / RATED_PAYLOAD_MOMENT_NM);
  }

  snapshot(): WorldSnapshot {
    const state = this.state;
    const anchors = exactAnchors(state.robot.joints);
    return {
      schemaVersion: SCENE_SCHEMA_VERSION,
      sceneId: CANONICAL_SCENE_ID,
      seed: this.seed,
      turn: state.actionCount,
      simulationTimeS: round9(state.simulationTimeS),
      robot: {
        ...state.robot,
        joints: [...state.robot.joints],
        command: { ...state.robot.command },
        anchors: {
          base: cloneVec3(anchors.base),
          shoulder: cloneVec3(anchors.shoulder),
          elbow: cloneVec3(anchors.elbow),
          wrist: cloneVec3(anchors.wrist),
          tcp: cloneVec3(anchors.tcp),
        },
      },
      object: {
        ...state.object,
        position: cloneVec3(state.object.position),
        size: cloneVec3(state.object.size),
      },
      barrier: sceneBox(
        state.barrier.id,
        state.barrier.center,
        state.barrier.size,
      ),
      target: sceneBox(state.target.id, state.target.center, state.target.size),
      lastAction: state.lastAction,
      events: state.lastEvents.map(cloneEvent),
      eventLog: state.eventLog.map(cloneEvent),
      success: state.success,
      terminal: state.success || state.levelFailed,
    };
  }
}

export function phaseForTurn(turn: number, success: boolean): string {
  if (success) return "Placement verified";
  if (turn < 3) return "Selecting height channel";
  if (turn < 18) return "Descending to bilateral grasp";
  if (turn < 33) return "Lifting with verified attachment";
  if (turn < 43) return "Cleared carry around barrier";
  return "Controlled descent into target";
}
