import {
  BARRIER_CAP_OVERHANG_M,
  BARRIER_CAP_THICKNESS_M,
  BASE_COLUMN_RADIUS_M,
  BASE_HEIGHT_M,
  FOREARM_RADIUS_M,
  GRIPPER_JAW_RADIAL_OFFSET_M,
  GRIPPER_JAW_SIZE_M,
  GRIPPER_JAW_VERTICAL_OFFSET_M,
  GRIPPER_PALM_RADIAL_OFFSET_M,
  GRIPPER_PALM_SIZE_M,
  TARGET_WALL_HEIGHT_M,
  TARGET_WALL_THICKNESS_M,
  UPPER_ARM_RADIUS_M,
  WORKCELL_POST_SIZE_M,
  WORKCELL_POST_X_M,
  WORKCELL_POST_Y_M,
  WORKCELL_REAR_WALL_CENTER_M,
  WORKCELL_REAR_WALL_SIZE_M,
  WRIST_LINK_RADIUS_M,
  cloneVec3,
  type ArmAnchors,
  type SceneBox,
  type Vec3,
} from "./model";

export interface Capsule {
  id: string;
  start: Vec3;
  end: Vec3;
  radius: number;
}

export interface YawBox {
  id: string;
  center: Vec3;
  size: Vec3;
  yaw: number;
}

export function barrierCap(barrier: SceneBox): SceneBox {
  return {
    id: "barrier-cap",
    center: [
      barrier.center[0],
      barrier.center[1],
      barrier.center[2] +
        barrier.size[2] * 0.5 +
        BARRIER_CAP_THICKNESS_M * 0.5,
    ],
    size: [
      barrier.size[0] + BARRIER_CAP_OVERHANG_M * 2,
      barrier.size[1] + BARRIER_CAP_OVERHANG_M * 2,
      BARRIER_CAP_THICKNESS_M,
    ],
  };
}

export function targetWalls(target: SceneBox): SceneBox[] {
  const halfX = target.size[0] * 0.5;
  const halfY = target.size[1] * 0.5;
  const centerZ = TARGET_WALL_HEIGHT_M * 0.5;
  return [
    {
      id: "target-wall-y-minus",
      center: [target.center[0], target.center[1] - halfY, centerZ],
      size: [
        target.size[0],
        TARGET_WALL_THICKNESS_M,
        TARGET_WALL_HEIGHT_M,
      ],
    },
    {
      id: "target-wall-y-plus",
      center: [target.center[0], target.center[1] + halfY, centerZ],
      size: [
        target.size[0],
        TARGET_WALL_THICKNESS_M,
        TARGET_WALL_HEIGHT_M,
      ],
    },
    {
      id: "target-wall-x-minus",
      center: [target.center[0] - halfX, target.center[1], centerZ],
      size: [
        TARGET_WALL_THICKNESS_M,
        target.size[1],
        TARGET_WALL_HEIGHT_M,
      ],
    },
    {
      id: "target-wall-x-plus",
      center: [target.center[0] + halfX, target.center[1], centerZ],
      size: [
        TARGET_WALL_THICKNESS_M,
        target.size[1],
        TARGET_WALL_HEIGHT_M,
      ],
    },
  ];
}

export function workcellSolids(): SceneBox[] {
  return [
    {
      id: "workcell-rear-wall",
      center: cloneVec3(WORKCELL_REAR_WALL_CENTER_M),
      size: cloneVec3(WORKCELL_REAR_WALL_SIZE_M),
    },
    ...WORKCELL_POST_X_M.map((x, index) => ({
      id: `workcell-safety-post-${index + 1}`,
      center: [
        x,
        WORKCELL_POST_Y_M,
        WORKCELL_POST_SIZE_M[2] * 0.5,
      ] as Vec3,
      size: cloneVec3(WORKCELL_POST_SIZE_M),
    })),
  ];
}

export function robotCapsules(anchors: ArmAnchors): Capsule[] {
  return [
    {
      id: "base-column",
      start: [0, 0, BASE_HEIGHT_M],
      end: cloneVec3(anchors.shoulder),
      radius: BASE_COLUMN_RADIUS_M,
    },
    {
      id: "upper-arm",
      start: cloneVec3(anchors.shoulder),
      end: cloneVec3(anchors.elbow),
      radius: UPPER_ARM_RADIUS_M,
    },
    {
      id: "forearm",
      start: cloneVec3(anchors.elbow),
      end: cloneVec3(anchors.wrist),
      radius: FOREARM_RADIUS_M,
    },
    {
      id: "wrist-link",
      start: cloneVec3(anchors.wrist),
      end: cloneVec3(anchors.tcp),
      radius: WRIST_LINK_RADIUS_M,
    },
  ];
}

function offsetPoint(
  origin: Vec3,
  yaw: number,
  radial: number,
  tangential: number,
  vertical: number,
): Vec3 {
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  return [
    origin[0] + radial * cosine - tangential * sine,
    origin[1] + radial * sine + tangential * cosine,
    origin[2] + vertical,
  ];
}

export function gripperBoxes(
  tcp: Vec3,
  yaw: number,
  aperture: number,
): YawBox[] {
  return [
    {
      id: "gripper-palm",
      center: offsetPoint(tcp, yaw, GRIPPER_PALM_RADIAL_OFFSET_M, 0, 0),
      size: cloneVec3(GRIPPER_PALM_SIZE_M),
      yaw,
    },
    ...([-1, 1] as const).map((direction) => ({
      id: `gripper-jaw-${direction < 0 ? "minus" : "plus"}`,
      center: offsetPoint(
        tcp,
        yaw,
        GRIPPER_JAW_RADIAL_OFFSET_M,
        aperture * 0.5 * direction,
        GRIPPER_JAW_VERTICAL_OFFSET_M,
      ),
      size: cloneVec3(GRIPPER_JAW_SIZE_M),
      yaw,
    })),
  ];
}

export function attachmentLocalOffset(
  tcp: Vec3,
  yaw: number,
  objectCenter: Vec3,
): Vec3 {
  const deltaX = objectCenter[0] - tcp[0];
  const deltaY = objectCenter[1] - tcp[1];
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  return [
    deltaX * cosine + deltaY * sine,
    -deltaX * sine + deltaY * cosine,
    objectCenter[2] - tcp[2],
  ];
}

export function attachedWorldPosition(
  tcp: Vec3,
  yaw: number,
  localOffset: Vec3,
): Vec3 {
  return offsetPoint(
    tcp,
    yaw,
    localOffset[0],
    localOffset[1],
    localOffset[2],
  );
}
