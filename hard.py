

# 질문
'''
내가 안돌려봤음. GPT가 와꾸 짬. gpt나 선규나 나한테 물어보셈

외부 D405로부터 
1. 로봇 위치 칼리브레이션 (april tag를 로봇 3cm 앞에 둘 예정임) 
2. 물체 세그멘테이션 (SAM2나 yolo8seg) 
3. 세그멘테이션 정보 바탕으로 pcd 세그멘테이션 
4. pcd 세그멘테이션 정보 바탕으로 pcd중심점 포즈 계산(로봇 기준) 코드 짜줄래? 
윈도우, Python3.10'''

# 답변
'''
tag 크기, TCP–tag 오프셋, Euler 순서 등은 너가 직접 실험하면서 맞춰야됨
pip install pyrealsense2 opencv-python numpy pupil-apriltags ultralytics
'''

"""
RGB-D D405 + AprilTag + YOLOv8-seg + PCD Center in Robot Frame (Windows, Python 3.10)

Pipeline:
1) Calibrate camera-to-robot using an AprilTag placed 3 cm in front of the robot TCP.
2) Segment the target object in the RGB image using YOLOv8-seg.
3) Use the segmentation mask to extract a point cloud subset from the depth image.
4) Compute the 3D centroid of the segmented PCD in the camera frame.
5) Transform the centroid into the robot base frame using the calibrated extrinsic.

Assumptions:
- D405 is fixed externally (eye-to-hand).
- AprilTag: tag36h11, known physical size (TAG_SIZE_M).
- Robot provides TCP pose in base frame as [x(mm), y(mm), z(mm), rx(deg), ry(deg), rz(deg)].
- TCP → Tag transform (3 cm "in front") is known and approximated as a pure translation.
"""

import time
import math
from typing import Tuple, Optional

import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from ultralytics import YOLO

# ---------------------------------------------------------------------------
#  Helper: SE(3) transforms
# ---------------------------------------------------------------------------

def euler_xyz_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Convert XYZ (Rx, Ry, Rz in degrees) to 3x3 rotation matrix.
    NOTE: Order here is Rz * Ry * Rx (common convention); adjust if needed.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
#  RealSense wrapper
# ---------------------------------------------------------------------------

class D405Camera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(self.config)

        # Align depth to color
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # Get intrinsics (color-aligned depth)
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        depth_profile = depth_frame.profile.as_video_stream_profile()
        self.depth_intr = depth_profile.get_intrinsics()  # fx, fy, ppx, ppy, etc.

        self.color_shape = (color_frame.height, color_frame.width)

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray, rs.video_stream_profile]:
        """Return (color_bgr, depth_image_in_meters, depth_frame_profile)."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get frames from D405.")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth * depth_scale

        return color, depth_m, depth_frame.profile.as_video_stream_profile()

    def deproject_pixel(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """Convert (u, v, depth) to 3D point in camera coordinates (meters)."""
        point = rs.rs2_deproject_pixel_to_point(self.depth_intr, [u, v], depth_m)
        return np.array(point)  # [X, Y, Z] in camera frame

    def stop(self):
        self.pipeline.stop()


# ---------------------------------------------------------------------------
#  AprilTag-based camera–robot calibration
# ---------------------------------------------------------------------------

class AprilTagCalibrator:
    def __init__(self, tag_size_m: float, family: str = "tag36h11"):
        self.tag_size_m = tag_size_m
        self.detector = Detector(
            families=family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def detect_tag_pose(
        self,
        gray: np.ndarray,
        camera_params: Tuple[float, float, float, float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (R_cam_tag, t_cam_tag) if tag is detected, else None."""
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=self.tag_size_m,
        )
        if len(tags) == 0:
            return None

        # Use the first detected tag
        tag = tags[0]
        R = tag.pose_R  # 3x3
        t = tag.pose_t  # (3, 1) or (3,)
        return R, t.reshape(3)

    def calibrate_base_to_cam(
        self,
        camera: D405Camera,
        robot_tcp_pose: np.ndarray,
        tcp_to_tag_offset: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute T_base_cam using:
          - robot_tcp_pose: [x(mm), y(mm), z(mm), rx(deg), ry(deg), rz(deg)]
          - tcp_to_tag_offset: [tx, ty, tz] in meters (tag position in TCP frame)
          - AprilTag pose from camera.

        Returns:
            4x4 T_base_cam or None if tag was not detected.
        """
        color, depth, _ = camera.get_frame()
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        fx = camera.depth_intr.fx
        fy = camera.depth_intr.fy
        cx = camera.depth_intr.ppx
        cy = camera.depth_intr.ppy
        camera_params = (fx, fy, cx, cy)

        pose = self.detect_tag_pose(gray, camera_params)
        if pose is None:
            print("[WARN] AprilTag not detected. Calibration failed.")
            return None

        R_cam_tag, t_cam_tag = pose
        T_cam_tag = make_transform(R_cam_tag, t_cam_tag)

        # Robot: base <- tcp
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = robot_tcp_pose
        t_base_tcp = np.array([x_mm, y_mm, z_mm]) / 1000.0  # mm -> m
        R_base_tcp = euler_xyz_to_matrix(rx_deg, ry_deg, rz_deg)
        T_base_tcp = make_transform(R_base_tcp, t_base_tcp)

        # TCP <- tag (pure translation)
        T_tcp_tag = make_transform(np.eye(3), tcp_to_tag_offset)

        # base <- cam = base<-tcp * tcp<-tag * (cam<-tag)^-1
        T_base_cam = T_base_tcp @ T_tcp_tag @ invert_transform(T_cam_tag)
        return T_base_cam


# ---------------------------------------------------------------------------
#  YOLOv8 segmentation + PCD centroid
# ---------------------------------------------------------------------------

class SegmentationPCDProcessor:
    def __init__(self, model_path: str = "yolov8n-seg.pt", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def get_class_id(self, class_name: str) -> int:
        """Find YOLO class id from its name (COCO, etc.)."""
        for cid, name in self.model.names.items():
            if name == class_name:
                return int(cid)
        raise ValueError(f"Class name '{class_name}' not found in model.names.")

    def segment(self, bgr: np.ndarray, target_cls_name: str) -> Optional[np.ndarray]:
        """
        Run YOLOv8-seg on the image and return a binary mask (H, W) for the target class.
        Returns None if no instance is found.
        """
        target_id = self.get_class_id(target_cls_name)

        # retina_masks=True ensures masks are in original image size. 
        results = self.model(
            bgr,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )[0]

        if results.masks is None:
            return None

        masks = results.masks.data.cpu().numpy()  # (N, H, W)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Combine all masks of desired class into one binary mask
        H, W = masks.shape[1:]
        combined = np.zeros((H, W), dtype=np.uint8)

        for mask, cid in zip(masks, cls_ids):
            if cid == target_id:
                combined = np.logical_or(combined, mask > 0.5)

        if not combined.any():
            return None

        return combined.astype(np.uint8)

    @staticmethod
    def compute_centroid_from_mask(
        mask: np.ndarray,
        depth_m: np.ndarray,
        camera: D405Camera,
        max_points: int = 5000,
    ) -> Optional[np.ndarray]:
        """
        From a binary mask (H, W) and aligned depth image, compute 3D centroid
        in camera frame. Returns [X, Y, Z] in meters or None.
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        # Optionally subsample pixels for speed
        if len(xs) > max_points:
            idx = np.random.choice(len(xs), size=max_points, replace=False)
            xs = xs[idx]
            ys = ys[idx]

        points = []
        for u, v in zip(xs, ys):
            z = depth_m[v, u]
            if z <= 0:
                continue
            pt_cam = camera.deproject_pixel(u, v, float(z))
            points.append(pt_cam)

        if len(points) == 0:
            return None

        points = np.vstack(points)  # (N, 3)
        centroid = np.mean(points, axis=0)
        return centroid  # [X, Y, Z] in camera frame (m)


# ---------------------------------------------------------------------------
#  Main example
# ---------------------------------------------------------------------------

def main():
    # ----------------------------
    # User parameters
    # ----------------------------
    TAG_SIZE_M = 0.05        # 5 cm tag (adjust to your print size)
    TCP_TO_TAG = np.array([0.03, 0.0, 0.0])  # 3 cm in front along TCP +X (edit as needed)
    TARGET_CLASS = "bottle"  # for COCO; change to your object class
    YOLO_DEVICE = "cuda"     # "cuda" if you have NVIDIA GPU, otherwise "cpu"

    # Here you should integrate your real robot API.
    # For this template, we assume you can query TCP pose as:
    #   [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    # For now, we use a dummy pose (robot at origin, no rotation).
    def get_robot_tcp_pose_dummy():
        return np.array([0.0, 0.0, 0.3 * 1000, 0.0, 0.0, 0.0])  # 0.3 m in Z

    # ----------------------------
    # Initialize devices
    # ----------------------------
    cam = D405Camera(width=640, height=480, fps=30)
    calibrator = AprilTagCalibrator(TAG_SIZE_M)
    seg_pcd = SegmentationPCDProcessor(device=YOLO_DEVICE)

    # ----------------------------
    # 1) Calibration (single-shot)
    # ----------------------------
    print("[INFO] Place AprilTag 3 cm in front of robot TCP and press ENTER for calibration.")
    input()

    tcp_pose = get_robot_tcp_pose_dummy()
    T_base_cam = calibrator.calibrate_base_to_cam(
        camera=cam,
        robot_tcp_pose=tcp_pose,
        tcp_to_tag_offset=TCP_TO_TAG,
    )

    if T_base_cam is None:
        print("[ERROR] Calibration failed. Exiting.")
        cam.stop()
        return

    print("[INFO] Calibration done. T_base_cam =")
    print(T_base_cam)

    # ----------------------------
    # 2–4) Online loop: segmentation + PCD centroid + transform
    # ----------------------------
    print("[INFO] Starting main loop. Press ESC to exit.")

    try:
        while True:
            color, depth_m, _ = cam.get_frame()

            # 2) RGB segmentation
            mask = seg_pcd.segment(color, TARGET_CLASS)
            if mask is None:
                cv2.imshow("RGB", color)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                continue

            # 3) PCD centroid in camera frame
            centroid_cam = seg_pcd.compute_centroid_from_mask(mask, depth_m, cam)
            if centroid_cam is None:
                cv2.imshow("RGB", color)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                continue

            # 4) Transform to robot base frame
            p_cam_h = np.hstack([centroid_cam, 1.0])
            p_base_h = T_base_cam @ p_cam_h
            centroid_base = p_base_h[:3]

            print(
                f"[INFO] Object centroid:"
                f" cam = {centroid_cam[0]:.3f}, {centroid_cam[1]:.3f}, {centroid_cam[2]:.3f} [m]  |  "
                f"base = {centroid_base[0]:.3f}, {centroid_base[1]:.3f}, {centroid_base[2]:.3f} [m]"
            )

            # Visualization: overlay mask
            vis = color.copy()
            vis[mask.astype(bool)] = (0, 255, 0)
            cv2.imshow("Segmentation (green = target)", vis)
            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
