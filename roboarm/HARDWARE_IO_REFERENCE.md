# Reference hardware and I/O boundary

This simulator targets the Waveshare RoArm-M2-S sold by Reichelt together with
a separately connected Logitech C920s. The two devices do not share a hardware
clock or trigger.

## What the arm actually reports

The stock controller accepts newline-delimited JSON over 115200-baud USB/UART,
or JSON in HTTP requests over Wi-Fi. The documented `{"T":105}` query returns a
`T=1051` object with:

- firmware-derived end-effector `x`, `y`, `z` in millimetres;
- base, shoulder, elbow, and clamp/wrist encoder angles `b/s/e/t` in radians;
- signed bus-servo load readings `torB/torS/torE/torH`;
- four torque-enable flags; and
- supply voltage `v` in 0.01 V units.

The joints use 12-bit (4096-count) magnetic encoders, a nominal no-load speed
of 40 rpm, and have approximately ±4 mm one-direction repeatability under the
same load. The controller-derived XYZ values are not an independent external
position sensor.

There is no documented stock feedback field for object identity, attachment,
jaw aperture in metres, TCP contact force, collision reason, or goal state.
The board includes an IMU and current/voltage hardware and the servos support
richer low-level status, but those values are not silently treated as fields in
the stock T=1051 packet.

Primary references:

- https://www.waveshare.com/product/roarm-m2-s.htm
- https://www.waveshare.com/wiki/RoArm-M2-S_Robotic_Arm_Control
- https://www.waveshare.com/wiki/RoArm-M2-S_JSON_Command_Meaning

## What the separate webcam reports

The C920s is a USB/UVC camera. Logitech documents 1920×1080 at 30 fps or
1280×720 at 30 fps, fixed 78° diagonal field of view, autofocus, automatic
light correction, and stereo microphones. It does not provide depth, robot
joint state, force, or a hardware-synchronized arm timestamp.

The simulated connector models a 1920×1080 MJPG source at 30 fps, decodes RGB,
and aspect-preserving downsamples it to the public 128×72×3 RGB8 observation.
Audio is intentionally not part of the task observation. The renderer is an
explicit deterministic pinhole approximation because Logitech does not publish
unit-specific lens calibration coefficients.

Primary reference:

- https://www.logitech.com/en-us/shop/p/c920s-pro-hd-webcam.960-001257

## Pairing and connector semantics

Arm feedback and webcam frames are host-timestamped independently. A sample
pairs one T=1051 response with the newest camera frame available at response
receipt, records both timestamps and the skew, and rejects a stale frame. This
is temporal correlation, not synchronization.

`roboarm_game.device_io.ReferenceDeviceIO` is the physical transport boundary.
It accepts small serial-JSON and UVC adapter protocols so deployment may use
pyserial plus V4L2/OpenCV without making either dependency part of the
deterministic simulator. `RoboArmConnector` applies the same pairing checks to
simulated evidence and hashes the image and feedback both separately and as a
pair.

Host collision interlocks remain connector decisions. Public telemetry says
only whether a command was interlocked; a simulator collision category is not
misrepresented as a physical arm sensor reading.
