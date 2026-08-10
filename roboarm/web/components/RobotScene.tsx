"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

import { exactAnchors } from "@/lib/kinematics";
import {
  BASE_COLUMN_RADIUS_M,
  BASE_HEIGHT_M,
  BASE_RADIUS_M,
  ELBOW_JOINT_RADIUS_M,
  FOREARM_RADIUS_M,
  GRIPPER_JAW_RADIAL_OFFSET_M,
  GRIPPER_JAW_SIZE_M,
  GRIPPER_JAW_VERTICAL_OFFSET_M,
  GRIPPER_PALM_RADIAL_OFFSET_M,
  GRIPPER_PALM_SIZE_M,
  SHOULDER_JOINT_RADIUS_M,
  TABLE_SIZE_M,
  UPPER_ARM_RADIUS_M,
  WRIST_JOINT_RADIUS_M,
  WRIST_LINK_RADIUS_M,
  type JointVector,
  type Vec3,
  type WorldSnapshot,
} from "@/lib/model";
import {
  barrierCap,
  targetWalls,
  workcellSolids,
} from "@/lib/physicalGeometry";

interface RobotSceneProps {
  snapshot: WorldSnapshot;
}

interface Segment {
  mesh: THREE.Mesh<THREE.CylinderGeometry, THREE.MeshStandardMaterial>;
  radius: number;
}

function worldVector(value: Vec3): THREE.Vector3 {
  return new THREE.Vector3(value[0], value[2], -value[1]);
}

function setWorldPosition(object: THREE.Object3D, value: Vec3): void {
  object.position.set(value[0], value[2], -value[1]);
}

function makeMaterial(
  color: THREE.ColorRepresentation,
  metalness: number,
  roughness: number,
  extras: Partial<THREE.MeshStandardMaterialParameters> = {},
): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color,
    metalness,
    roughness,
    ...extras,
  });
}

function box(
  size: [number, number, number],
  material: THREE.MeshStandardMaterial,
): THREE.Mesh<THREE.BoxGeometry, THREE.MeshStandardMaterial> {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(size[0], size[1], size[2]),
    material,
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function segment(
  radius: number,
  material: THREE.MeshStandardMaterial,
  radialSegments = 32,
): Segment {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, 1, radialSegments, 1, false),
    material,
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return { mesh, radius };
}

const UP = new THREE.Vector3(0, 1, 0);
function updateSegment(
  value: Segment,
  start: Vec3,
  end: Vec3,
): void {
  const from = worldVector(start);
  const to = worldVector(end);
  const direction = to.clone().sub(from);
  const length = Math.max(direction.length(), 1e-6);
  value.mesh.position.copy(from.add(to).multiplyScalar(0.5));
  value.mesh.scale.set(1, length, 1);
  value.mesh.quaternion.setFromUnitVectors(UP, direction.normalize());
}

function makeLabel(
  text: string,
  foreground: string,
  background: string,
): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("2D canvas unavailable");
  context.fillStyle = background;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = foreground;
  context.lineWidth = 5;
  context.strokeRect(8, 8, canvas.width - 16, canvas.height - 16);
  context.fillStyle = foreground;
  context.font = "700 54px ui-monospace, SFMono-Regular, monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2 + 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
    }),
  );
  sprite.scale.set(0.22, 0.055, 1);
  return sprite;
}

function addTargetBin(
  scene: THREE.Scene,
  snapshot: WorldSnapshot,
): {
  beacon: THREE.PointLight;
  ring: THREE.Mesh<THREE.TorusGeometry, THREE.MeshStandardMaterial>;
} {
  const target = snapshot.target;
  const center = worldVector(target.center);
  const cyan = makeMaterial("#22c6b9", 0.35, 0.3, {
    transparent: true,
    opacity: 0.58,
    emissive: new THREE.Color("#0b514e"),
    emissiveIntensity: 0.35,
  });
  const floor = box([target.size[0], target.size[2], target.size[1]], cyan);
  floor.position.copy(center);
  scene.add(floor);

  for (const wall of targetWalls(target)) {
    const mesh = box(
      [wall.size[0], wall.size[2], wall.size[1]],
      cyan,
    );
    setWorldPosition(mesh, wall.center);
    scene.add(mesh);
  }

  const ringMaterial = makeMaterial("#43f0d8", 0.15, 0.25, {
    emissive: new THREE.Color("#0b6e65"),
    emissiveIntensity: 0.8,
  });
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.078, 0.003, 12, 64),
    ringMaterial,
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(center.x, 0.006, center.z);
  scene.add(ring);

  const beacon = new THREE.PointLight("#48f0d6", 0, 0.45, 2);
  beacon.position.set(center.x, 0.12, center.z);
  scene.add(beacon);
  return { beacon, ring };
}

function addWorkcell(scene: THREE.Scene, snapshot: WorldSnapshot): void {
  const floorMaterial = makeMaterial("#242b2d", 0.05, 0.9);
  const floor = box([2.4, 0.045, 2.2], floorMaterial);
  floor.position.y = -0.105;
  scene.add(floor);

  const tableMaterial = makeMaterial("#4d5758", 0.62, 0.24);
  const table = box(
    [TABLE_SIZE_M[0], TABLE_SIZE_M[2], TABLE_SIZE_M[1]],
    tableMaterial,
  );
  table.position.y = -TABLE_SIZE_M[2] * 0.5;
  scene.add(table);

  const edgeMaterial = makeMaterial("#161c1f", 0.75, 0.25);
  for (const [x, z] of [
    [-0.41, -0.41],
    [0.41, -0.41],
    [-0.41, 0.41],
    [0.41, 0.41],
  ]) {
    const leg = box([0.045, 0.42, 0.045], edgeMaterial);
    leg.position.set(x, -0.245, z);
    scene.add(leg);
  }

  const grid = new THREE.GridHelper(0.9, 18, "#556366", "#3d484a");
  grid.position.y = 0.001;
  (grid.material as THREE.Material).opacity = 0.42;
  (grid.material as THREE.Material).transparent = true;
  scene.add(grid);

  const [rearWallSolid, ...postSolids] = workcellSolids();
  const rearWall = box(
    [
      rearWallSolid.size[0],
      rearWallSolid.size[2],
      rearWallSolid.size[1],
    ],
    makeMaterial("#252d30", 0.08, 0.82),
  );
  setWorldPosition(rearWall, rearWallSolid.center);
  scene.add(rearWall);

  for (let index = -3; index <= 3; index += 1) {
    const rail = box(
      [0.016, 0.62, 0.018],
      makeMaterial(index % 2 === 0 ? "#374144" : "#30383b", 0.25, 0.6),
    );
    rail.position.set(index * 0.19, 0.33, -0.608);
    scene.add(rail);
  }

  const header = makeLabel("ROBOT CELL · RB01", "#a9f8e9", "#162123");
  header.position.set(-0.18, 0.54, -0.604);
  scene.add(header);

  const barrierMaterial = makeMaterial("#a53c32", 0.3, 0.34, {
    emissive: new THREE.Color("#3a0906"),
    emissiveIntensity: 0.18,
  });
  const barrier = box(
    [
      snapshot.barrier.size[0],
      snapshot.barrier.size[2],
      snapshot.barrier.size[1],
    ],
    barrierMaterial,
  );
  setWorldPosition(barrier, snapshot.barrier.center);
  scene.add(barrier);

  const capSolid = barrierCap(snapshot.barrier);
  const cap = box(
    [capSolid.size[0], capSolid.size[2], capSolid.size[1]],
    makeMaterial("#f3b33b", 0.15, 0.35, {
      emissive: new THREE.Color("#73430c"),
      emissiveIntensity: 0.25,
    }),
  );
  setWorldPosition(cap, capSolid.center);
  scene.add(cap);

  addTargetBin(scene, snapshot);

  for (const solid of postSolids) {
    const post = box(
      [solid.size[0], solid.size[2], solid.size[1]],
      makeMaterial("#687679", 0.8, 0.2),
    );
    setWorldPosition(post, solid.center);
    scene.add(post);
  }
}

function createJoint(
  radius: number,
  color: THREE.ColorRepresentation,
): THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial> {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 32, 20),
    makeMaterial(color, 0.72, 0.22),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export default function RobotScene({ snapshot }: RobotSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const snapshotRef = useRef(snapshot);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    const mount = mountRef.current;
    if (mount === null) return;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.domElement.setAttribute("data-testid", "rgb-camera-canvas");
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#12191c");
    scene.fog = new THREE.Fog("#12191c", 1.3, 2.8);

    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 5);
    camera.position.set(0.72, 0.5, 0.1);
    camera.lookAt(0.14, 0.13, -0.07);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.14, 0.13, -0.07);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 0.55;
    controls.maxDistance = 1.05;
    controls.minPolarAngle = 0.35;
    controls.maxPolarAngle = 1.4;
    controls.minAzimuthAngle = -1.45;
    controls.maxAzimuthAngle = 1.45;
    controls.enablePan = false;

    const pmrem = new THREE.PMREMGenerator(renderer);
    const environment = pmrem.fromScene(new RoomEnvironment(), 0.04);
    scene.environment = environment.texture;

    const hemisphere = new THREE.HemisphereLight("#d9f1ef", "#313b3e", 1.15);
    scene.add(hemisphere);
    const key = new THREE.DirectionalLight("#fff1dc", 4.2);
    key.position.set(0.55, 0.9, 0.45);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -0.8;
    key.shadow.camera.right = 0.8;
    key.shadow.camera.top = 0.8;
    key.shadow.camera.bottom = -0.8;
    scene.add(key);
    const rim = new THREE.SpotLight("#4fc8ff", 22, 2.2, 0.55, 0.6, 1.2);
    rim.position.set(-0.55, 0.65, 0.5);
    rim.target.position.set(0.15, 0.12, -0.05);
    scene.add(rim, rim.target);

    addWorkcell(scene, snapshotRef.current);
    const targetCenter = worldVector(snapshotRef.current.target.center);
    const targetRing = scene.children.find(
      (child) => child instanceof THREE.Mesh && child.geometry instanceof THREE.TorusGeometry,
    ) as THREE.Mesh<THREE.TorusGeometry, THREE.MeshStandardMaterial> | undefined;
    const targetBeacon = scene.children.find(
      (child) => child instanceof THREE.PointLight && child.color.getHexString() === "48f0d6",
    ) as THREE.PointLight | undefined;

    const baseMaterial = makeMaterial("#26383f", 0.78, 0.2);
    const armMaterial = makeMaterial("#e98225", 0.52, 0.24, {
      emissive: new THREE.Color("#351404"),
      emissiveIntensity: 0.11,
    });
    const forearmMaterial = makeMaterial("#a94a21", 0.65, 0.22);
    const jointMaterial = makeMaterial("#172126", 0.86, 0.16);
    const gripperMaterial = makeMaterial("#dce4e2", 0.88, 0.17);

    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(
        BASE_RADIUS_M,
        BASE_RADIUS_M,
        BASE_HEIGHT_M,
        48,
      ),
      baseMaterial,
    );
    base.position.set(0, BASE_HEIGHT_M * 0.5, 0);
    base.castShadow = true;
    base.receiveShadow = true;
    scene.add(base);
    const baseRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.052, 0.004, 12, 64),
      makeMaterial("#61c8d8", 0.42, 0.22, {
        emissive: new THREE.Color("#174e57"),
        emissiveIntensity: 0.7,
      }),
    );
    baseRing.rotation.x = -Math.PI / 2;
    baseRing.position.y = 0.037;
    scene.add(baseRing);

    const column = segment(BASE_COLUMN_RADIUS_M, baseMaterial);
    const upperArm = segment(UPPER_ARM_RADIUS_M, armMaterial);
    const forearm = segment(FOREARM_RADIUS_M, forearmMaterial);
    const wristLink = segment(WRIST_LINK_RADIUS_M, jointMaterial);
    scene.add(column.mesh, upperArm.mesh, forearm.mesh, wristLink.mesh);

    const shoulderJoint = createJoint(SHOULDER_JOINT_RADIUS_M, "#27353a");
    const elbowJoint = createJoint(ELBOW_JOINT_RADIUS_M, "#253036");
    const wristJoint = createJoint(WRIST_JOINT_RADIUS_M, "#d7dfde");
    scene.add(shoulderJoint, elbowJoint, wristJoint);

    const gripper = new THREE.Group();
    const palm = box(
      [
        GRIPPER_PALM_SIZE_M[0],
        GRIPPER_PALM_SIZE_M[2],
        GRIPPER_PALM_SIZE_M[1],
      ],
      gripperMaterial,
    );
    palm.position.x = GRIPPER_PALM_RADIAL_OFFSET_M;
    gripper.add(palm);
    const jawSize: [number, number, number] = [
      GRIPPER_JAW_SIZE_M[0],
      GRIPPER_JAW_SIZE_M[2],
      GRIPPER_JAW_SIZE_M[1],
    ];
    const jawLeft = box(jawSize, gripperMaterial);
    const jawRight = box(jawSize, gripperMaterial);
    jawLeft.position.set(
      GRIPPER_JAW_RADIAL_OFFSET_M,
      GRIPPER_JAW_VERTICAL_OFFSET_M,
      -0.04,
    );
    jawRight.position.set(
      GRIPPER_JAW_RADIAL_OFFSET_M,
      GRIPPER_JAW_VERTICAL_OFFSET_M,
      0.04,
    );
    gripper.add(jawLeft, jawRight);
    const tcpMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.007, 16, 12),
      makeMaterial("#62f1da", 0.15, 0.2, {
        emissive: new THREE.Color("#28cbb4"),
        emissiveIntensity: 1.7,
      }),
    );
    gripper.add(tcpMarker);
    scene.add(gripper);

    const workpieceMaterial = makeMaterial("#f3c54f", 0.44, 0.25, {
      emissive: new THREE.Color("#4d3103"),
      emissiveIntensity: 0.13,
    });
    const object = box(
      [
        snapshotRef.current.object.size[0],
        snapshotRef.current.object.size[2],
        snapshotRef.current.object.size[1],
      ],
      workpieceMaterial,
    );
    scene.add(object);

    const objectTop = box(
      [
        snapshotRef.current.object.size[0] * 0.72,
        0.002,
        snapshotRef.current.object.size[1] * 0.72,
      ],
      makeMaterial("#171d1f", 0.32, 0.4),
    );
    scene.add(objectTop);

    const carryLineGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]);
    const carryLine = new THREE.Line(
      carryLineGeometry,
      new THREE.LineDashedMaterial({
        color: "#6be7d8",
        dashSize: 0.015,
        gapSize: 0.01,
        transparent: true,
        opacity: 0.52,
      }),
    );
    carryLine.computeLineDistances();
    scene.add(carryLine);

    let currentJoints = [...snapshotRef.current.robot.joints] as JointVector;
    const currentObject = worldVector(snapshotRef.current.object.position);
    let currentAperture = snapshotRef.current.robot.gripperAperture;
    let lastTurn = snapshotRef.current.turn;
    let pulseStart = performance.now();
    let lastFrameTime = performance.now();

    const resize = (): void => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();

    let animationId = 0;
    const animate = (now: number): void => {
      const elapsed = Math.min(0.05, Math.max(0, (now - lastFrameTime) / 1000));
      lastFrameTime = now;
      const target = snapshotRef.current;
      if (target.turn !== lastTurn) {
        lastTurn = target.turn;
        pulseStart = now;
      }

      const smoothing = 1 - Math.exp(-elapsed * 13);
      currentJoints = currentJoints.map(
        (value, index) =>
          value + (target.robot.joints[index] - value) * smoothing,
      ) as JointVector;
      currentAperture +=
        (target.robot.gripperAperture - currentAperture) * smoothing;
      currentObject.lerp(worldVector(target.object.position), smoothing);

      const anchors = exactAnchors(currentJoints);
      updateSegment(column, [0, 0, BASE_HEIGHT_M], anchors.shoulder);
      updateSegment(upperArm, anchors.shoulder, anchors.elbow);
      updateSegment(forearm, anchors.elbow, anchors.wrist);
      updateSegment(wristLink, anchors.wrist, anchors.tcp);
      setWorldPosition(shoulderJoint, anchors.shoulder);
      setWorldPosition(elbowJoint, anchors.elbow);
      setWorldPosition(wristJoint, anchors.wrist);
      setWorldPosition(gripper, anchors.tcp);
      gripper.rotation.y = target.robot.command.azimuth;
      jawLeft.position.z = -currentAperture * 0.5;
      jawRight.position.z = currentAperture * 0.5;

      object.position.copy(currentObject);
      objectTop.position.copy(currentObject);
      objectTop.position.y += target.object.size[2] * 0.5 + 0.001;

      const positions = carryLineGeometry.attributes.position as THREE.BufferAttribute;
      const tcp = worldVector(anchors.tcp);
      positions.setXYZ(0, tcp.x, tcp.y, tcp.z);
      positions.setXYZ(1, currentObject.x, currentObject.y, currentObject.z);
      positions.needsUpdate = true;
      carryLine.visible = target.object.attached;
      carryLine.computeLineDistances();

      const pulse = Math.max(0, 1 - (now - pulseStart) / 900);
      const hasContact = target.events.some((event) =>
        event.kind.includes("contact") || event.kind === "object_attached",
      );
      workpieceMaterial.emissiveIntensity = target.success
        ? 0.9
        : hasContact
          ? 0.13 + pulse * 1.2
          : 0.13;
      armMaterial.emissiveIntensity = target.robot.rejected
        ? 0.11 + pulse * 1.5
        : 0.11;
      baseRing.material.emissiveIntensity =
        0.55 + 0.2 * Math.sin(now * 0.003);
      if (targetRing !== undefined) {
        targetRing.material.emissiveIntensity = target.success
          ? 1.5 + 0.5 * Math.sin(now * 0.008)
          : 0.65;
        targetRing.rotation.z = target.success ? now * 0.0008 : 0;
      }
      if (targetBeacon !== undefined) {
        targetBeacon.intensity = target.success ? 4.5 + Math.sin(now * 0.01) : 0;
        targetBeacon.position.set(targetCenter.x, 0.12, targetCenter.z);
      }

      controls.update();
      renderer.render(scene, camera);
      animationId = requestAnimationFrame(animate);
    };
    animationId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationId);
      observer.disconnect();
      controls.dispose();
      environment.texture.dispose();
      pmrem.dispose();
      scene.traverse((node) => {
        if (node instanceof THREE.Mesh || node instanceof THREE.Line) {
          node.geometry?.dispose();
          const materials = Array.isArray(node.material)
            ? node.material
            : [node.material];
          for (const material of materials) {
            if (material instanceof THREE.SpriteMaterial) {
              material.map?.dispose();
            }
            material.dispose();
          }
        }
        if (node instanceof THREE.Sprite) {
          node.material.map?.dispose();
          node.material.dispose();
        }
      });
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className="robot-scene"
      data-testid="rgb-camera"
      aria-label="Interactive simulated RGB camera view of the robotic workcell"
    />
  );
}
