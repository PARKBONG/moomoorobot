# api_multi.py

import os
import json
import cv2
import math
import numpy as np
from pupil_apriltags import Detector


class AprilTagTracker:
    """
    하나의 카메라로 여러 AprilTag를 추적하는 클래스.
    
    사용자가 쓰는 함수는 주로 하나:
        get_pose(ID)

    사용 예:
        tracker = AprilTagTracker(pop_window=True, ...)
        tracker.add_tag(5, 2.0)  # ID=5, 크기 2cm
        pose = tracker.get_pose(5)
        print(pose)  # [x, y, z, rx_deg, ry_deg, rz_deg]
    """

    def __init__(
        self,
        pop_window=False,
        tag_family="tag36h11",
        camera_index=0,
        intrinsic_path=None,
        extrinsic_path=None,
        window_name="AprilTag Pose Viewer",
    ):
        self.pop_window = bool(pop_window)
        self.tag_family = tag_family
        self.camera_index = camera_index
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        self.window_name = window_name

        # 등록된 태그 정보 {id: size_m}
        self.tags = {}

        # camera & extrinsic & detector
        self.K_orig = None          # 보정 시 사용된 원래 K
        self.K = None               # 현재 사용 중인 K (getOptimalNewCameraMatrix 결과)
        self.dist_coeffs = None
        self.camera_params = None
        self.calib_image_size = None  # (w, h)

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
    # 태그 관리
    # ---------------------------------------------------------
    def add_tag(self, ID, size_cm):
        """추적할 태그 등록 (ID: 정수, size_cm: cm 단위)"""
        self.tags[int(ID)] = float(size_cm) / 100.0  # cm → m

    def remove_tag(self, ID):
        """태그 제거"""
        ID = int(ID)
        if ID in self.tags:
            del self.tags[ID]

    # ---------------------------------------------------------
    # 내부 로드 함수들
    # ---------------------------------------------------------
    def _resolve_path(self, path):
        cur = os.path.abspath(__file__)
        d = os.path.dirname(cur)
        return os.path.join(d, path)

    def _load_camera_params(self):
        if self.intrinsic_path is None:
            raise ValueError("intrinsic_path 가 없습니다.")
        json_path = self._resolve_path(self.intrinsic_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        intr = data["intrinsics"]
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        self.K_orig = K.copy()
        self.K = K.copy()

        self.dist_coeffs = np.array(data["dist_coeffs"], float)
        self.camera_params = (fx, fy, cx, cy)

        # 보정에 사용된 해상도(있으면) 저장
        if "image_size" in data:
            w, h = data["image_size"]
            self.calib_image_size = (int(w), int(h))
            print(f"[INFO] 보정에 사용된 이미지 해상도: {w} x {h}")

    def _load_extrinsic_params(self):
        if self.extrinsic_path is None:
            raise ValueError("extrinsic_path 가 없습니다.")
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
            raise RuntimeError("카메라 열기 실패")

        # 보정에 사용한 해상도에 맞추어 해상도 설정 시도
        if self.calib_image_size is not None:
            calib_w, calib_h = self.calib_image_size
            print(f"[INFO] 보정 해상도에 맞추어 카메라 해상도 설정 시도: {calib_w} x {calib_h}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_h)

        # 실제 해상도 확인
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 실제 카메라 해상도: {w} x {h}")

        if self.calib_image_size is not None and (w != self.calib_image_size[0] or h != self.calib_image_size[1]):
            print("[WARN] 실제 카메라 해상도가 보정에 사용된 해상도와 다릅니다. "
                  "Intrinsic과 정확히 일치하지 않을 수 있습니다.")

        # getOptimalNewCameraMatrix 로 새 K 계산
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K_orig, self.dist_coeffs, (w, h), 1)
        self.K = new_K
        print("[INFO] getOptimalNewCameraMatrix로 얻은 새 K:")
        print(self.K)

        # camera_params도 새 K 기준으로 업데이트
        new_fx = float(self.K[0, 0])
        new_fy = float(self.K[1, 1])
        new_cx = float(self.K[0, 2])
        new_cy = float(self.K[1, 2])
        self.camera_params = (new_fx, new_fy, new_cx, new_cy)

    # ---------------------------------------------------------
    # 기본 수학 함수
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
        sy = -R[2, 0]
        sy = np.clip(sy, -1.0, 1.0)
        pitch = math.asin(sy)
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    # ---------------------------------------------------------
    # 시각화용 함수들
    # ---------------------------------------------------------
    def _draw_world_origin_axes(self, frame, axis_len=0.2):
        """
        월드(로봇 베이스) 원점과 XYZ 축을 카메라 영상 위에 표시.
        """
        R_cw = self.T_camera_world[:3, :3]
        t_cw = self.T_camera_world[:3, 3]

        origin = t_cw
        axes = [
            ("X_w", R_cw[:, 0], (0, 0, 255)),   # X: 빨강
            ("Y_w", R_cw[:, 1], (0, 255, 0)),   # Y: 초록
            ("Z_w", R_cw[:, 2], (255, 0, 0)),   # Z: 파랑
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

    def _draw_line_with_global_text(self, frame, pos_cam, pos_world, tag_id):
        """
        로봇 베이스(월드 원점)에서 태그까지 선을 그리고,
        선 위에 Global 좌표, 태그 근처에 CAM 좌표 표시.
        """
        o_uv = self._project_point(self.K, self.origin_cam)
        p_uv = self._project_point(self.K, pos_cam)
        if not o_uv or not p_uv:
            return

        # 선 그리기 (회색)
        cv2.line(frame, o_uv, p_uv, (127, 127, 127), 2)

        # 선 중간 위치
        mid = ((o_uv[0] + p_uv[0]) // 2, (o_uv[1] + p_uv[1]) // 2)

        text_g = f"ID {tag_id}: (%.2f, %.2f, %.2f)" % tuple(pos_world)
        cv2.putText(frame, text_g, (mid[0] + 5, mid[1] - 5), 0, 0.6, (127, 127, 127), 2)

    # ---------------------------------------------------------
    # 내부 연산: 모든 태그 pose 계산
    # ---------------------------------------------------------
    def _compute_all_poses(self):
        """
        detect()는 한 번만 호출하고,
        등록된 태그 ID만 pose 계산.
        반환값: { ID: [float...], ... }
        """
        if self.cap is None or not self.cap.isOpened():
            return {}
    
        ret, frame_raw = self.cap.read()
        if not ret:
            return {}

        # 보정 시 K_orig, dist 기준으로 undistort하고,
        # 결과 이미지는 self.K (new_K) 기준.
        frame = cv2.undistort(frame_raw, self.K_orig, self.dist_coeffs, None, self.K)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=1.0,  # scaling later
        )

        results = {}

        for det in detections:
            tag_id = det.tag_id
            if tag_id not in self.tags:
                continue

            size_m = self.tags[tag_id]

            R_ct = np.array(det.pose_R)
            t_ct = np.array(det.pose_t).reshape(3) * size_m

            pos_cam = t_ct
            pos_world_vec = self.R_wc @ pos_cam + self.t_wc

            R_wt = self.R_wc @ R_ct
            roll, pitch, yaw = self._rotation_matrix_to_rpy(R_wt)

            x = float(pos_world_vec[0])
            y = float(pos_world_vec[1])
            z = float(pos_world_vec[2])
            rx = float(math.degrees(roll))
            ry = float(math.degrees(pitch))
            rz = float(math.degrees(yaw))

            results[tag_id] = [x, y, z, rx, ry, rz]

            # 시각화: 태그별 선/텍스트
            if self.pop_window:
                self._draw_line_with_global_text(frame, pos_cam, [x, y, z], tag_id)

        # 시각화: 월드 축, 안내 텍스트, imshow
        if self.pop_window:
            self._draw_world_origin_axes(frame)
            cv2.putText(
                frame,
                "ESC: close window",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            cv2.imshow(self.window_name, frame)

        return results

    # ---------------------------------------------------------
    # public API: 단일 태그 pose만 반환
    # ---------------------------------------------------------
    def get_pose(self, ID):
        """
        ID에 해당하는 태그 pose를 Python float 리스트로 반환.
        못 찾으면 None.
        """
        ID = int(ID)
        poses = self._compute_all_poses()
        return poses.get(ID, None)

    # ---------------------------------------------------------
    # 종료 처리
    # ---------------------------------------------------------
    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def is_pressed_esc(self):
        """
        pop_window=True일 때만 작동.
        ESC 또는 'q'가 눌리면 True, 아니면 False 반환.
        """
        if not self.pop_window:
            return False

        key = cv2.waitKey(1) & 0xFF

        # ESC 또는 q 키
        if key == 27 or key == ord('q'):
            self.close()
            return True
        
        return False
