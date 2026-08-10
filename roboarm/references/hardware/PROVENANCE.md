# RoArm-M2-S hardware provenance

**Captured:** 2026-07-30
**Role:** immutable engineering inputs for later kinematics work; not runtime
dependencies and not Phase-0 physics.

## Pinned official sources

| Purpose | Repository | Revision | Path |
|---|---|---|---|
| ROS/Xacro joint chain | `waveshareteam/roarm_ws` | `40dbd84b553695212fab713e8465f817ba95454d` | `src/roarm_main/roarm_description/urdf/roarm_m2/roarm_m2.xacro` |
| Firmware/product context | `waveshareteam/roarm_m2` | `c6ccc5bda2eb92df2f0850d3e63cc42b81557f4f` | repository root |

The first repository’s default branch was `ros2-humble` when pinned. The second
repository’s default branch was `main`. Full commit identifiers, rather than
branch names, are normative.

The preserved [roarm_m2.xacro](roarm_m2.xacro) is the byte-for-byte reference
captured from:

```text
https://raw.githubusercontent.com/waveshareteam/roarm_ws/40dbd84b553695212fab713e8465f817ba95454d/src/roarm_main/roarm_description/urdf/roarm_m2/roarm_m2.xacro
```

Its SHA-256 is:

```text
4144e6f20919554755c7dd515f7a236a9aa128fc5507bede3b66cba4ee751c2c
```

[transforms.json](transforms.json) is a literal extraction of every joint
origin, axis, parent/child pair, and joint limit from that preserved file. Tests
parse the Xacro and compare every extracted lexical value to the fixture.

## Product-envelope cross-check

The specification also cites the Reichelt `WS-25974` product page for
manufacturer-level envelope claims (4 DOF, stated reach/load/repeatability,
servo and interface descriptions). That page is mutable and therefore is a
descriptive cross-check, not a source for exact transform constants:

```text
https://www.reichelt.com/ch/en/shop/product/robot_arm_kit_roarm-m2-s-4_dof-405729
```

No ROS package, Xacro processor, vendor SDK, or firmware package is installed or
imported by `roboarm_game`.
