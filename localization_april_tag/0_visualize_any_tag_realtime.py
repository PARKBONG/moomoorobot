import os
import json
import cv2
import numpy as np
from pupil_apriltags import Detector

# ============================================================
# 설정 부분 (사용자가 바꿔도 되는 값)
# ============================================================

# 1) 카메라 보정 결과(JSON) 파일 경로
#    - 아래 JSON 예시처럼 fx, fy, cx, cy, dist_coeffs 가 들어있는 파일
#    - 예: calib_realsense_d405.json
# 현재 파일(0_visualize_any_tag_realtime.py) 기준 상대 경로임!
CALIB_JSON_PATH = "./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json"

# 2) 사용하려는 AprilTag 패밀리 이름
#    - tag36h11 체커보드를 썼다면 "tag36h11"
TAG_FAMILY = "tag36h11"

# 3) 태그 한 변의 실제 길이 (미터 단위)
#    - 예) 20 cm 태그라면 0.20
TAG_SIZE_M = 0.02

# 4) 카메라 인덱스
#    - 웹캠이 하나이면 보통 0
CAMERA_INDEX = 0

# 설정 부분 끝 =================================================


def load_camera_params(json_path):
    """
    카메라 보정 결과(JSON 파일)를 읽어서
    카메라 내부 파라미터(K)와 왜곡 계수(D)를 반환하는 함수.
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

    dist_coeffs = np.array(data["dist_coeffs"], dtype=float)  # (k1, k2, p1, p2, k3, ...)

    # 3x3 카메라 내부 행렬 K
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # pupil_apriltags Detector에 넣어줄 카메라 파라미터 튜플
    camera_params = (fx, fy, cx, cy)

    # 이미지 해상도 정보가 있으면 참고용으로 출력
    if "image_size" in data:
        w, h = data["image_size"]
        print(f"[INFO] 보정에 사용된 이미지 해상도: {w} x {h}")

    return K, dist_coeffs, camera_params


def create_detector():
    """
    AprilTag 검출기를 만드는 함수.
    설정은 크게 건드릴 필요 없음.
    """
    detector = Detector(
        families=TAG_FAMILY,   # 사용할 Tag 패밀리 (예: tag36h11)
        nthreads=4,            # CPU 스레드 수
        quad_decimate=1.0,     # 이미지 축소 비율(속도 vs 정확도)
        quad_sigma=0.0,        # 블러 정도
        refine_edges=1,        # 테두리 정교화
        decode_sharpening=0.25,
        debug=0,
    )
    return detector


def project_point(K, p_cam):
    """
    카메라 좌표계의 3D 점을 이미지(픽셀) 좌표로 투영하는 함수.

    K      : 3x3 카메라 내부 행렬
    p_cam  : (3,) 또는 (3,1) 형태의 3D 점 (단위: m)
    return : (u, v) (픽셀 좌표), 만약 Z<=0 이면 None
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


def main():
    # --- 카메라 보정 파라미터 로드 -----------------------------------------
    print("[INFO] 카메라 보정 파라미터(JSON) 불러오는 중...")
    K, dist_coeffs, camera_params = load_camera_params(CALIB_JSON_PATH)
    print("[INFO] K (카메라 내부 행렬):")
    print(K)
    print("[INFO] 왜곡 계수(dist_coeffs):", dist_coeffs)

    # --- AprilTag 검출기 생성 ----------------------------------------------
    detector = create_detector()

    # --- 카메라 열기 -------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. CAMERA 연결 또는 INDEX를 확인하세요.")
        return

    print("[INFO] ESC 키를 누르면 종료합니다.")

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽어오지 못했습니다.")
            break

        # 1) 왜곡 보정 (undistort)
        frame = cv2.undistort(frame_raw, K, dist_coeffs)

        # 2) 흑백 이미지로 변환 (검출기 입력용)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3) AprilTag 검출 및 자세 추정
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE_M,
        )

        # 검출된 태그가 하나도 없으면 안내 출력
        if len(detections) == 0:
            cv2.putText(
                frame,
                "No tags detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # 각 태그에 대해 화면에 표시 및 위치 출력
            for det in detections:
                tag_id = det.tag_id

                # R: 회전 행렬 (3x3), t: 태그 중심 위치 (3x1, 카메라 좌표계 기준, 단위: m)
                R = np.array(det.pose_R)
                t = np.array(det.pose_t)

                # --- 1) 태그 테두리(초록색 네모) 그리기 ----------------------
                corners = np.array(det.corners, dtype=np.int32)  # (4, 2)
                corners = corners.reshape(-1, 1, 2)
                cv2.polylines(
                    frame,
                    [corners],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

                # --- 2) 태그 중심 위치에 점 찍기 --------------------------
                c_x, c_y = det.center
                center_px = (int(c_x), int(c_y))
                cv2.circle(frame, center_px, 4, (0, 255, 0), -1)

                # --- 3) 태그의 로컬 좌표축(X,Y,Z) 그리기 -------------------
                axis_len = 0.04  # 4 cm 정도 길이
                origin_cam = t.reshape(3)  # 태그 중심(카메라 좌표계)

                # Z축이 카메라를 향하는지 확인 (정상적인 경우 Z축은 카메라 쪽을 향함)
                z_axis_direction = R[:, 2]  # 태그의 Z축 방향 벡터
                
                # 태그 중심에서 카메라 방향으로의 벡터 (단위 벡터로 정규화)
                to_camera = origin_cam / np.linalg.norm(origin_cam)

                dot_product = np.dot(z_axis_direction, to_camera)

                x_axis_cam = origin_cam + R[:, 0] * axis_len
                y_axis_cam = origin_cam + R[:, 1] * axis_len
                z_axis_cam = origin_cam + R[:, 2] * axis_len

                o_uv = project_point(K, origin_cam)
                x_uv = project_point(K, x_axis_cam)
                y_uv = project_point(K, y_axis_cam)
                z_uv = project_point(K, z_axis_cam)

                if o_uv is not None:
                    # X축: 빨강
                    if x_uv is not None:
                        cv2.line(frame, o_uv, x_uv, (0, 0, 255), 2)
                    # Y축: 초록
                    if y_uv is not None:
                        cv2.line(frame, o_uv, y_uv, (0, 255, 0), 2)
                    # Z축: 파랑
                    if z_uv is not None:
                        z_color = (255, 0, 0) if dot_product > 0 else (0, 165, 255)  # 파랑 or 주황
                        cv2.line(frame, o_uv, z_uv, z_color, 2)

                # --- 4) 태그 ID 및 (x,y,z) 위치 텍스트로 표시 --------------
                x_m, y_m, z_m = t.flatten()
                text_id = f"ID: {tag_id}"
                text_pos = f"({x_m:.3f},{y_m:.3f},{z_m:.3f})"

                # ID는 태그 위쪽에
                cv2.putText(
                    frame,
                    text_id,
                    (center_px[0] + 5, center_px[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,  # thickness를 2로 증가 (bold 효과)
                    cv2.LINE_AA,
                )

                # 좌표는 태그 아래쪽에 (글씨 작게)
                cv2.putText(
                    frame,
                    text_pos,
                    (center_px[0] + 5, center_px[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,  # thickness를 2로 증가 (bold 효과)
                    cv2.LINE_AA,
                )

        # 4) 결과 화면 보여주기
        cv2.imshow("AprilTag Real Time Visualizer", frame)

        # 5) ESC 키를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
