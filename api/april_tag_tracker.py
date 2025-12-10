# api.py

import os
import json
import cv2
import math
import numpy as np
from pupil_apriltags import Detector


class AprilTagTracker:
    """
    AprilTag 하나의 위치/자세를 쉽게 얻기 위한 간단한 클래스.
    get_pose()는 [x, y, z, rx_deg, ry_deg, rz_deg] 반환.

    사용 예시)
        from api import AprilTagTracker

        tag = AprilTagTracker(
            ID=5,
            size_cm=7.0,
            pop_window=True,
            tag_family="tag36h11",
            camera_index=0,
            intrinsic_path="./intrinsic_calibration_result.json",
            extrinsic_path="./extrinsic_calibration_result.json",
        )

        pose = tag.get_pose()  # [x, y, z, roll, pitch, yaw]
        print(pose)
    """

    def __init__(
        self,
        ID,
        size_cm,
        pop_window=True,
        tag_family="tag36h11",
        camera_index=0,
        intrinsic_path=None,
        extrinsic_path=None,
        window_name="AprilTag Pose Viewer",
    ):
        
        """
        ID           : 인식할 AprilTag ID (정수)
        size_cm      : 태그 한 변 길이 (cm 단위)
        pop_window   : True이면 화면 시각화 (카메라 영상, 선/텍스트 표시)
        tag_family   : AprilTag 패밀리 이름 (예: "tag36h11")
        camera_index : 카메라 인덱스 (기본 0)
        intrinsic_path : 카메라 내부 파라미터 JSON 경로
        extrinsic_path : 카메라-월드(로봇 베이스) 외부 파라미터 JSON 경로
        """
        
        self.tag_id = int(ID)
        self.tag_size_m = float(size_cm) / 100.0  # cm → m
        self.pop_window = bool(pop_window)
        self.tag_family = tag_family
        self.camera_index = camera_index
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        self.window_name = window_name

        self.K = None
        self.dist_coeffs = None
        self.camera_params = None
        self.T_camera_world = None
        self.T_world_camera = None
        self.R_wc = None
        self.t_wc = None
        self.origin_cam = None
        self.detector = None
        self.cap = None

        # 초기화
        self._load_camera_params()
        self._load_extrinsic_params()
        self._create_detector()
        self._open_camera()

    # ---------------------------------------------------------
    # 내부 파일/초기화
    # ---------------------------------------------------------
    def _resolve_path(self, path):
        cur = os.path.abspath(__file__)
        d = os.path.dirname(cur)
        return os.path.join(d, path)

    def _load_camera_params(self):
        if self.intrinsic_path is None:
            raise ValueError("intrinsic_path 가 지정되지 않았습니다.")
        json_path = self._resolve_path(self.intrinsic_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        intr = data["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        self.dist_coeffs = np.array(data["dist_coeffs"], float)
        self.camera_params = (fx, fy, cx, cy)

    def _load_extrinsic_params(self):
        if self.extrinsic_path is None:
            raise ValueError("extrinsic_path 가 지정되지 않았습니다.")
        json_path = self._resolve_path(self.extrinsic_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.T_camera_world = np.array(data["T_camera_world"], float)
        self.T_world_camera = np.array(data["T_world_camera"], float)

        self.R_wc = self.T_world_camera[:3, :3]
        self.t_wc = self.T_world_camera[:3, 3]
        self.origin_cam = self.T_camera_world[:3, 3]

    def _create_detector(self):
        self.detector = Detector(
            families=self.tag_family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def _open_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")

    # ---------------------------------------------------------
    # 기본 수학 함수들
    # ---------------------------------------------------------
    @staticmethod
    def _project_point(K, p_cam):
        X, Y, Z = p_cam
        if Z <= 0:
            return None
        u = K[0, 0] * X / Z + K[0, 2]
        v = K[1, 1] * Y / Z + K[1, 2]
        return int(u), int(v)

    @staticmethod
    def _rotation_matrix_to_rpy(R):
        """
        회전행렬 → roll, pitch, yaw (라디안)
        """
        sy = -R[2, 0]
        sy = np.clip(sy, -1.0, 1.0)
        pitch = math.asin(sy)
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    # ---------------------------------------------------------
    # 시각화 함수
    # ---------------------------------------------------------
    def _draw_world_origin_axes(self, frame, axis_len=0.2):
        R_cw = self.T_camera_world[:3, :3]
        t_cw = self.T_camera_world[:3, 3]

        origin = t_cw
        axes = [
            ("X_w", R_cw[:, 0], (0, 0, 255)),
            ("Y_w", R_cw[:, 1], (0, 255, 0)),
            ("Z_w", R_cw[:, 2], (255, 0, 0)),
        ]

        o_uv = self._project_point(self.K, origin)
        if o_uv is None:
            return

        for name, axis_vec, color in axes:
            p = origin + axis_vec * axis_len
            p_uv = self._project_point(self.K, p)
            if p_uv:
                cv2.line(frame, o_uv, p_uv, color, 2)
                cv2.putText(frame, name, p_uv, 0, 0.5, color, 1)

    def _draw_line_with_global_text(self, frame, pos_cam, pos_world):
        o_uv = self._project_point(self.K, self.origin_cam)
        p_uv = self._project_point(self.K, pos_cam)
        if not o_uv or not p_uv:
            return

        cv2.line(frame, o_uv, p_uv, (127, 127, 127), 2)

        mid = ((o_uv[0] + p_uv[0]) // 2, (o_uv[1] + p_uv[1]) // 2)

        text_g = "Global: (%.2f, %.2f, %.2f)" % tuple(pos_world)
        cv2.putText(frame, text_g, (mid[0] + 5, mid[1] - 5), 0, 0.6, (0, 0, 0), 2)

        text_c = "CAM: (%.2f, %.2f, %.2f)" % tuple(pos_cam)
        cv2.putText(frame, text_c, (p_uv[0] + 5, p_uv[1] + 15), 0, 0.6, (0, 0, 0), 2)

    # ---------------------------------------------------------
    # 주요 public 메서드
    # ---------------------------------------------------------
    def get_pose(self):
        """
        AprilTag ID의 pose를 읽고
        [x, y, z, rx_deg, ry_deg, rz_deg] 반환.
        찾지 못하면 None.
        """
        ret, frame_raw = self.cap.read()
        if not ret:
            return None

        frame = cv2.undistort(frame_raw, self.K, self.dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size_m,
        )

        det = next((d for d in detections if d.tag_id == self.tag_id), None)
        if det is None:
            if self.pop_window:
                cv2.putText(frame, "Tag not found", (30, 40), 0, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
            return None

        R_ct = np.array(det.pose_R)
        t_ct = np.array(det.pose_t).reshape(3)

        pos_cam = t_ct
        pos_world = self.R_wc @ pos_cam + self.t_wc

        # 회전 변환 (월드 기준)
        R_wt = self.R_wc @ R_ct
        roll, pitch, yaw = self._rotation_matrix_to_rpy(R_wt)

        # ★ 요청 반영: degree 변환
        rx_deg = math.degrees(roll)
        ry_deg = math.degrees(pitch)
        rz_deg = math.degrees(yaw)

        if self.pop_window:
            self._draw_world_origin_axes(frame)
            self._draw_line_with_global_text(frame, pos_cam, pos_world)
            cv2.putText(frame, "ESC to close", (30, 30), 0, 0.7, (0, 0, 0), 2)
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.close()

        return [pos_world[0], pos_world[1], pos_world[2], rx_deg, ry_deg, rz_deg]

    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        try:
            self.close()
        except:
            pass