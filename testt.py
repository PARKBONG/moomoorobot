import time

from api.april_tag_tracker import AprilTagTracker

object_handle = AprilTagTracker(
    ID=5,
    size_cm=2.0,
    pop_window=False,
    intrinsic_path="./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json",
    extrinsic_path="./../localization_april_tag/extrinsic_calibration_result.json",
)

target_handle = AprilTagTracker(
    ID=6,
    size_cm=6.0,
    pop_window=False,
    intrinsic_path="./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json",
    extrinsic_path="./../localization_april_tag/extrinsic_calibration_result.json",
)

while True:
    object_pose = object_handle.get_pose()
    if object_pose is not None:
        x, y, z, rx, ry, rz = object_pose
        print(
            "[Object Pose] x: %.3f m, y: %.3f m, z: %.3f m, rx: %.3f deg, ry: %.3f deg, rz: %.3f deg"
            % (x, y, z, rx, ry, rz)
        )
        time.sleep(0.1)
        
    target_pose = target_handle.get_pose()
    if target_pose is not None:
        x, y, z, rx, ry, rz = target_pose
        print(
            "[Target Pose] x: %.3f m, y: %.3f m, z: %.3f m, rx: %.3f deg, ry: %.3f deg, rz: %.3f deg"
            % (x, y, z, rx, ry, rz)
        )
        time.sleep(0.01)