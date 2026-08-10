import type { SceneBox, Vec3 } from "./model";

export function expandedContains(
  box: SceneBox,
  point: Vec3,
  margin: number,
): boolean {
  return point.every(
    (value, index) =>
      Math.abs(value - box.center[index]) <= box.size[index] * 0.5 + margin,
  );
}

export function segmentIntersectsBox(
  start: Vec3,
  end: Vec3,
  box: SceneBox,
  radius: number,
): boolean {
  const length = Math.hypot(
    end[0] - start[0],
    end[1] - start[1],
    end[2] - start[2],
  );
  const samples = Math.max(2, Math.ceil(length / Math.max(radius * 0.5, 0.004)));
  for (let index = 0; index <= samples; index += 1) {
    const ratio = index / samples;
    const point = start.map(
      (value, axis) => value + (end[axis] - value) * ratio,
    ) as Vec3;
    if (expandedContains(box, point, radius)) {
      return true;
    }
  }
  return false;
}

export function sphereIntersectsBox(
  center: Vec3,
  radius: number,
  box: SceneBox,
  margin = 0,
): boolean {
  let distanceSquared = 0;
  center.forEach((value, axis) => {
    const delta =
      Math.abs(value - box.center[axis]) - box.size[axis] * 0.5;
    if (delta > 0) distanceSquared += delta * delta;
  });
  return distanceSquared < (radius + margin) ** 2;
}

export function pointSegmentDistance(
  point: Vec3,
  start: Vec3,
  end: Vec3,
): number {
  const segment = start.map(
    (value, axis) => end[axis] - value,
  ) as Vec3;
  const relative = point.map(
    (value, axis) => value - start[axis],
  ) as Vec3;
  const lengthSquared = segment.reduce(
    (total, value) => total + value * value,
    0,
  );
  if (lengthSquared <= 1e-18) {
    return Math.hypot(
      point[0] - start[0],
      point[1] - start[1],
      point[2] - start[2],
    );
  }
  const ratio = Math.min(
    1,
    Math.max(
      0,
      relative.reduce(
        (total, value, axis) => total + value * segment[axis],
        0,
      ) / lengthSquared,
    ),
  );
  return Math.hypot(
    ...point.map(
      (value, axis) => value - (start[axis] + segment[axis] * ratio),
    ),
  );
}

export function segmentSegmentDistance(
  firstStart: Vec3,
  firstEnd: Vec3,
  secondStart: Vec3,
  secondEnd: Vec3,
): number {
  const first = firstStart.map(
    (value, axis) => firstEnd[axis] - value,
  ) as Vec3;
  const second = secondStart.map(
    (value, axis) => secondEnd[axis] - value,
  ) as Vec3;
  const offset = firstStart.map(
    (value, axis) => value - secondStart[axis],
  ) as Vec3;
  const dot = (left: Vec3, right: Vec3): number =>
    left.reduce((total, value, axis) => total + value * right[axis], 0);
  const firstLength = dot(first, first);
  const mixed = dot(first, second);
  const secondLength = dot(second, second);
  const firstOffset = dot(first, offset);
  const secondOffset = dot(second, offset);
  const denominator = firstLength * secondLength - mixed * mixed;
  const epsilon = 1e-15;

  if (firstLength <= epsilon && secondLength <= epsilon) {
    return Math.hypot(
      firstStart[0] - secondStart[0],
      firstStart[1] - secondStart[1],
      firstStart[2] - secondStart[2],
    );
  }
  if (firstLength <= epsilon) {
    return pointSegmentDistance(firstStart, secondStart, secondEnd);
  }
  if (secondLength <= epsilon) {
    return pointSegmentDistance(secondStart, firstStart, firstEnd);
  }

  let firstNumerator: number;
  let firstDenominator = denominator;
  let secondNumerator: number;
  let secondDenominator = denominator;
  if (denominator < epsilon) {
    firstNumerator = 0;
    firstDenominator = 1;
    secondNumerator = secondOffset;
    secondDenominator = secondLength;
  } else {
    firstNumerator = mixed * secondOffset - secondLength * firstOffset;
    secondNumerator = firstLength * secondOffset - mixed * firstOffset;
    if (firstNumerator < 0) {
      firstNumerator = 0;
      secondNumerator = secondOffset;
      secondDenominator = secondLength;
    } else if (firstNumerator > firstDenominator) {
      firstNumerator = firstDenominator;
      secondNumerator = secondOffset + mixed;
      secondDenominator = secondLength;
    }
  }

  if (secondNumerator < 0) {
    secondNumerator = 0;
    if (-firstOffset < 0) {
      firstNumerator = 0;
    } else if (-firstOffset > firstLength) {
      firstNumerator = firstDenominator;
    } else {
      firstNumerator = -firstOffset;
      firstDenominator = firstLength;
    }
  } else if (secondNumerator > secondDenominator) {
    secondNumerator = secondDenominator;
    if (-firstOffset + mixed < 0) {
      firstNumerator = 0;
    } else if (-firstOffset + mixed > firstLength) {
      firstNumerator = firstDenominator;
    } else {
      firstNumerator = -firstOffset + mixed;
      firstDenominator = firstLength;
    }
  }

  const firstRatio =
    Math.abs(firstNumerator) < epsilon
      ? 0
      : firstNumerator / firstDenominator;
  const secondRatio =
    Math.abs(secondNumerator) < epsilon
      ? 0
      : secondNumerator / secondDenominator;
  return Math.hypot(
    ...offset.map(
      (value, axis) =>
        value + firstRatio * first[axis] - secondRatio * second[axis],
    ),
  );
}

export function capsulesOverlap(
  firstStart: Vec3,
  firstEnd: Vec3,
  firstRadius: number,
  secondStart: Vec3,
  secondEnd: Vec3,
  secondRadius: number,
  margin = 0,
): boolean {
  return (
    segmentSegmentDistance(firstStart, firstEnd, secondStart, secondEnd) <
    firstRadius + secondRadius + margin
  );
}

export function segmentIntersectsVerticalCylinder(
  start: Vec3,
  end: Vec3,
  segmentRadius: number,
  centerX: number,
  centerY: number,
  cylinderRadius: number,
  bottom: number,
  top: number,
): boolean {
  const length = Math.hypot(
    end[0] - start[0],
    end[1] - start[1],
    end[2] - start[2],
  );
  const samples = Math.max(
    2,
    Math.ceil(length / Math.max(segmentRadius * 0.35, 0.003)),
  );
  const expandedRadius = cylinderRadius + segmentRadius;
  for (let index = 0; index <= samples; index += 1) {
    const ratio = index / samples;
    const point = start.map(
      (value, axis) => value + (end[axis] - value) * ratio,
    ) as Vec3;
    if (!(bottom - segmentRadius < point[2] && point[2] < top + segmentRadius)) {
      continue;
    }
    const radialSquared =
      (point[0] - centerX) ** 2 + (point[1] - centerY) ** 2;
    if (radialSquared < expandedRadius ** 2) return true;
  }
  return false;
}

export function sphereIntersectsVerticalCylinder(
  center: Vec3,
  radius: number,
  centerX: number,
  centerY: number,
  cylinderRadius: number,
  bottom: number,
  top: number,
): boolean {
  if (center[2] - radius >= top || center[2] + radius <= bottom) {
    return false;
  }
  return (
    Math.hypot(center[0] - centerX, center[1] - centerY) <
    cylinderRadius + radius
  );
}

export function yawBoxIntersectsBox(
  center: Vec3,
  size: Vec3,
  yaw: number,
  box: SceneBox,
  margin = 0,
): boolean {
  if (
    Math.abs(center[2] - box.center[2]) * 2 >=
    size[2] + box.size[2] + margin * 2
  ) {
    return false;
  }
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  const radial: [number, number] = [cosine, sine];
  const tangential: [number, number] = [-sine, cosine];
  const delta: [number, number] = [
    center[0] - box.center[0],
    center[1] - box.center[1],
  ];
  const firstHalf: [number, number] = [size[0] * 0.5, size[1] * 0.5];
  const secondHalf: [number, number] = [
    box.size[0] * 0.5,
    box.size[1] * 0.5,
  ];
  for (const axis of [
    [1, 0],
    [0, 1],
    radial,
    tangential,
  ] as [number, number][]) {
    const centerDistance = Math.abs(delta[0] * axis[0] + delta[1] * axis[1]);
    const firstRadius =
      firstHalf[0] *
        Math.abs(radial[0] * axis[0] + radial[1] * axis[1]) +
      firstHalf[1] *
        Math.abs(tangential[0] * axis[0] + tangential[1] * axis[1]);
    const secondRadius =
      secondHalf[0] * Math.abs(axis[0]) +
      secondHalf[1] * Math.abs(axis[1]);
    if (centerDistance >= firstRadius + secondRadius + margin) {
      return false;
    }
  }
  return true;
}

export function yawBoxIntersectsVerticalCylinder(
  center: Vec3,
  size: Vec3,
  yaw: number,
  centerX: number,
  centerY: number,
  cylinderRadius: number,
  bottom: number,
  top: number,
  margin = 0,
): boolean {
  if (
    center[2] - size[2] * 0.5 >= top + margin ||
    center[2] + size[2] * 0.5 <= bottom - margin
  ) {
    return false;
  }
  const deltaX = centerX - center[0];
  const deltaY = centerY - center[1];
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  const localRadial = deltaX * cosine + deltaY * sine;
  const localTangential = -deltaX * sine + deltaY * cosine;
  const outsideRadial = Math.max(Math.abs(localRadial) - size[0] * 0.5, 0);
  const outsideTangential = Math.max(
    Math.abs(localTangential) - size[1] * 0.5,
    0,
  );
  return (
    outsideRadial ** 2 + outsideTangential ** 2 <
    (cylinderRadius + margin) ** 2
  );
}

export function segmentIntersectsYawBox(
  start: Vec3,
  end: Vec3,
  segmentRadius: number,
  center: Vec3,
  size: Vec3,
  yaw: number,
): boolean {
  const length = Math.hypot(
    end[0] - start[0],
    end[1] - start[1],
    end[2] - start[2],
  );
  const samples = Math.max(
    2,
    Math.ceil(length / Math.max(segmentRadius * 0.35, 0.003)),
  );
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  const half = size.map((value) => value * 0.5 + segmentRadius) as Vec3;
  for (let index = 0; index <= samples; index += 1) {
    const ratio = index / samples;
    const point = start.map(
      (value, axis) => value + (end[axis] - value) * ratio,
    ) as Vec3;
    const deltaX = point[0] - center[0];
    const deltaY = point[1] - center[1];
    const local: Vec3 = [
      deltaX * cosine + deltaY * sine,
      -deltaX * sine + deltaY * cosine,
      point[2] - center[2],
    ];
    if (local.every((value, axis) => Math.abs(value) < half[axis])) {
      return true;
    }
  }
  return false;
}

export function boxesOverlap(
  firstCenter: Vec3,
  firstSize: Vec3,
  second: SceneBox,
  margin = 0,
): boolean {
  return firstCenter.every(
    (value, index) =>
      Math.abs(value - second.center[index]) * 2 <
      firstSize[index] + second.size[index] + 2 * margin,
  );
}

export function insideHorizontalTarget(
  center: Vec3,
  size: Vec3,
  target: SceneBox,
  margin = 0.004,
): boolean {
  return [0, 1].every(
    (axis) =>
      Math.abs(center[axis] - target.center[axis]) + size[axis] * 0.5 <=
      target.size[axis] * 0.5 - margin,
  );
}
