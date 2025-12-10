import os
import json
import cv2
import numpy as np
from pupil_apriltags import Detector

# ============================================================
# 설정 부분 (사용자가 바꿔도 되는 값)
# ============================================================

# 1) 카메라 보정 결과(JSON) 파일 경로
CALIB_JSON_PATH = "./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json"

# 2) Extrinsic 결과(JSON) 파일 경로
EXTRINSIC_JSON_PATH = "./extrinsic_calibration_result.json"

# 3) AprilTag 패밀리
TAG_FAMILY = "tag36h11"

# 4) Object / Target 태그 ID 및 크기 (m)
OBJECT_TAG_ID = 5
TARGET_TAG_ID = 6
OBJECT_TAG_SIZE_M = 0.02  # 필요에 맞게 수정
TARGET_TAG_SIZE_M = 0.06  # 필요에 맞게 수정

# 5) 카메라 인덱스
CAMERA_INDEX = 0

# 6) 윈도우 이름
WINDOW_NAME = "Object/Target Pose Viewer"

# ============================================================


def load_camera_params(json_path):
    """
    카메라 보정 결과(JSON 파일)를 읽어서
    카메라 내부 파라미터(K)와 왜곡 계수(D), camera_params를 반환.
    """
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    json_path = os.path.join(current_dir, json_path)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"카메라 보정 JSON 파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intr = data["intrinsics"]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    dist_coeffs = np.array(data["dist_coeffs"], dtype=float)

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    camera_params = (fx, fy, cx, cy)

    if "image_size" in data:
        w, h = data["image_size"]
        print(f"[INFO] 보정에 사용된 이미지 해상도: {w} x {h}")

    return K, dist_coeffs, camera_params


def load_extrinsic_params(json_path):
    """
    extrinsic_calibration_result.json 을 읽어서
    T_camera_world, T_world_camera 를 반환.
    """
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    json_path = os.path.join(current_dir, json_path)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Extrinsic JSON 파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    T_camera_world = np.array(data["T_camera_world"], dtype=float)
    T_world_camera = np.array(data["T_world_camera"], dtype=float)

    return T_camera_world, T_world_camera


def create_detector():
    """
    AprilTag 검출기 생성.
    """
    detector = Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    return detector


def project_point(K, p_cam):
    """
    카메라 좌표계의 3D 점 p_cam 을 이미지 픽셀 좌표로 투영.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    p = np.array(p_cam).reshape(3)
    X, Y, Z = float(p[0]), float(p[1]), float(p[2])

    if Z <= 0:
        return None

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return int(round(u)), int(round(v))


def draw_world_origin_axes(frame, K, T_camera_world, axis_len=0.2):
    """
    월드(=로봇 베이스) 원점과 XYZ 축을 카메라 영상 위에 그린다.
    T_camera_world: world -> camera (4x4)
    """
    R_cw = T_camera_world[0:3, 0:3]  # 3x3
    t_cw = T_camera_world[0:3, 3]    # (3,)

    origin_cam = t_cw
    # world 기준 단위벡터를 camera frame으로 표현: R_cw * e_i
    x_axis_cam = origin_cam + R_cw[:, 0] * axis_len
    y_axis_cam = origin_cam + R_cw[:, 1] * axis_len
    z_axis_cam = origin_cam + R_cw[:, 2] * axis_len

    o_uv = project_point(K, origin_cam)
    x_uv = project_point(K, x_axis_cam)
    y_uv = project_point(K, y_axis_cam)
    z_uv = project_point(K, z_axis_cam)

    if o_uv is None:
        return

    # X: 빨강, Y: 초록, Z: 파랑
    if x_uv is not None:
        cv2.line(frame, o_uv, x_uv, (0, 0, 255), 2)
        cv2.putText(frame, "X_w", x_uv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if y_uv is not None:
        cv2.line(frame, o_uv, y_uv, (0, 255, 0), 2)
        cv2.putText(frame, "Y_w", y_uv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if z_uv is not None:
        cv2.line(frame, o_uv, z_uv, (255, 0, 0), 2)
        cv2.putText(frame, "Z_w", z_uv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


def draw_line_with_global_text(frame, K, origin_cam, pos_cam, pos_world):
    """
    원점(origin_cam)에서 태그 위치(pos_cam)까지 빨간 선을 그리고,
    선 위에 Global: (x,y,z)를 검정색으로 표시.
    """
    o_uv = project_point(K, origin_cam)
    p_uv = project_point(K, pos_cam)

    if o_uv is None or p_uv is None:
        return

    # 빨간 선
    cv2.line(frame, o_uv, p_uv, (127, 127, 127), 2)

    # 선 중간 지점
    mid_uv = ((o_uv[0] + p_uv[0]) // 2, (o_uv[1] + p_uv[1]) // 2)

    text_global = "Global: (%.2f, %.2f, %.2f)" % (
        pos_world[0], pos_world[1], pos_world[2]
    )

    cv2.putText(
        frame,
        text_global,
        (mid_uv[0] + 5, mid_uv[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (127, 127, 127),   # 검정색
        2,
        cv2.LINE_AA,
    )


def main():
    # --- Intrinsic / Extrinsic 로드 ----------------------------------------
    print("[INFO] 카메라 Intrinsic 로드 중...")
    K, dist_coeffs, camera_params = load_camera_params(CALIB_JSON_PATH)
    print("[INFO] Extrinsic (T_camera_world, T_world_camera) 로드 중...")
    T_camera_world, T_world_camera = load_extrinsic_params(EXTRINSIC_JSON_PATH)

    print("[INFO] T_camera_world:")
    print(T_camera_world)
    print("[INFO] T_world_camera:")
    print(T_world_camera)

    # world <- camera
    R_wc = T_world_camera[0:3, 0:3]
    t_wc = T_world_camera[0:3, 3]

    # world origin in camera frame
    origin_cam = T_camera_world[0:3, 3]

    # --- AprilTag Detector -------------------------------------------------
    detector = create_detector()

    # 태그 크기 매핑
    tag_size_by_id = {
        OBJECT_TAG_ID: OBJECT_TAG_SIZE_M,
        TARGET_TAG_ID: TARGET_TAG_SIZE_M,
    }

    # detect() 에 넣을 기준 태그 크기
    tag_size_for_detect = float(OBJECT_TAG_SIZE_M)

    # --- 카메라 열기 -------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    print("[INFO] ESC: 종료")

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽어오지 못했습니다.")
            break

        # 왜곡 보정
        frame = cv2.undistort(frame_raw, K, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # AprilTag 검출
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size_for_detect,
        )

        # object/target의 위치(카메라 기준) 저장
        obj_cam = None
        tgt_cam = None

        # 태그 그리기 + pose 추출
        for det in detections:
            tag_id = det.tag_id

            if tag_id not in (OBJECT_TAG_ID, TARGET_TAG_ID):
                continue

            R_raw = np.array(det.pose_R)
            t_raw = np.array(det.pose_t)  # 기준 크기(tag_size_for_detect) 기준

            size_true = tag_size_by_id.get(tag_id, tag_size_for_detect)
            scale = size_true / tag_size_for_detect
            t_scaled = t_raw * scale  # 실제 태그 크기에 맞게 스케일 조정

            # 카메라 기준 위치
            pos_cam = t_scaled.reshape(3)

            if tag_id == OBJECT_TAG_ID:
                obj_cam = pos_cam
            elif tag_id == TARGET_TAG_ID:
                tgt_cam = pos_cam

            # 태그 테두리/센터
            corners = np.array(det.corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

            c_x, c_y = det.center
            center_px = (int(c_x), int(c_y))
            cv2.circle(frame, center_px, 4, (0, 255, 0), -1)

            # ID 표시
            text_id = f"ID: {tag_id}"
            cv2.putText(
                frame,
                text_id,
                (center_px[0] + 5, center_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # CAM: (x,y,z) (마커 근처, 흰색)
            text_cam = "CAM: (%.2f, %.2f, %.2f)" % (
                pos_cam[0], pos_cam[1], pos_cam[2]
            )
            cv2.putText(
                frame,
                text_cam,
                (center_px[0] + 5, center_px[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (127, 127, 127),
                2,
                cv2.LINE_AA,
            )

        # 월드(로봇 베이스) 원점 및 축 그리기
        draw_world_origin_axes(frame, K, T_camera_world, axis_len=0.2)

        # object / target에 대해 원점→태그까지 빨간 선 + Global 텍스트
        if obj_cam is not None:
            obj_world = R_wc @ obj_cam + t_wc
            draw_line_with_global_text(frame, K, origin_cam, obj_cam, obj_world)

        if tgt_cam is not None:
            tgt_world = R_wc @ tgt_cam + t_wc
            draw_line_with_global_text(frame, K, origin_cam, tgt_cam, tgt_world)

        # 화면 좌측 상단 안내
        cv2.putText(
            frame,
            "ESC: quit",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # 결과 표시
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
