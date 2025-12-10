import os
import json
import cv2
import numpy as np
import datetime
from pupil_apriltags import Detector

# ============================================================
# 설정 부분 (사용자가 바꿔도 되는 값)
# ============================================================

# 1) 카메라 보정 결과(JSON) 파일 경로
#    - fx, fy, cx, cy, dist_coeffs 가 들어있는 파일
# 현재 파일 기준 상대 경로
CALIB_JSON_PATH = "./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json"

# 2) 사용하려는 AprilTag 패밀리 이름
TAG_FAMILY = "tag36h11"

# 3) 로봇 베이스 모서리에 붙인 태그 ID (시계 방향)
#    예) 1 -> 2 -> 3 -> 4 순서로 시계 방향
CORNER_TAG_IDS = [1, 2, 3, 4]

# 4) 각 모서리 태그의 한 변 길이 (m)
#    CORNER_TAG_IDS 의 인덱스와 1:1 매칭
#    예) 태그 1: 0.20 m, 태그 2: 0.20 m, ...
CORNER_TAG_SIZES_M = [0.07, 0.07, 0.07, 0.07]

# 5) 카메라 인덱스
CAMERA_INDEX = 0

# 7) 저장 파일 이름 (고정)
JSON_FILENAME = "extrinsic_calibration_result.json"
IMG_FILENAME = "extrinsic_calibration_result.jpeg"

# 설정 부분 끝 =================================================

# 6) 윈도우에 표시될 이름
WINDOW_NAME = "AprilTag real time detection & extrinsic calibration"

# 8) 태그 좌표계와 로봇 베이스 좌표계의 축 매핑
# tag / robot
# +x / +x
# +y / -y
# +z / -z
#
# 로봇 베이스 프레임에서의 벡터 p_robot 를 태그 프레임 p_tag 로 표현할 때:
#   p_tag = R_ROBOT_TO_TAG @ p_robot
R_ROBOT_TO_TAG = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=float)

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


def compute_world_from_tags(tag_translations, tag_rotations):
    """
    로봇 베이스(월드 좌표계)를 정의하고,
    월드 좌표계에서 카메라 좌표계로 가는 변환을 계산하는 함수.

    입력:
      tag_translations : {tag_id: t(3x1)} 딕셔너리
                         t = 태그 중심 위치 (카메라 좌표계, 단위: m)
      tag_rotations    : {tag_id: R(3x3)} 딕셔너리
                         R = 태그 좌표 -> 카메라 좌표 회전행렬

    조건:
      - (1,3) 쌍 또는 (2,4) 쌍 중 하나 이상이 반드시 보여야 함.
      - 그렇지 않으면 ValueError 발생.

    월드(=로봇 베이스) 좌표계 정의:
      - 원점(origin):
        * (1,3)이 보이면 mid13 = (t1 + t3)/2
        * (2,4)가 보이면 mid24 = (t2 + t4)/2
        * 둘 다 보이면 origin = (mid13 + mid24)/2
        * 둘 중 하나만 보이면 그 중점만 사용
      - 축(orientation):
        * 현재 보이는 코너 태그들(1,2,3,4)의 회전행렬을 모두 평균낸 뒤,
          SVD를 이용해 SO(3)에 투영하여 태그축 기준 월드 회전 R_cam_world_tag 를 얻음.
        * 이후 R_ROBOT_TO_TAG 를 우측 곱하여
          로봇 베이스 축을 따르는 R_cam_world_robot 를 구성.
          (결과적으로 world = robot base frame)

    반환:
      T_camera_world : 4x4 행렬 (월드(로봇 베이스) -> 카메라)
                       p_cam = R * p_world + t
    """
    present = set(tag_translations.keys()) & set(CORNER_TAG_IDS)

    # (1,3) 쌍 또는 (2,4) 쌍이 하나라도 보여야 함
    has13 = (1 in present) and (3 in present)
    has24 = (2 in present) and (4 in present)

    if not (has13 or has24):
        raise ValueError("(1,3) 또는 (2,4) 태그 쌍 중 최소 하나는 보여야 합니다.")

    centers = []

    # (1,3) 대각선이 모두 보이는 경우
    if has13:
        t1 = tag_translations[1].reshape(3)
        t3 = tag_translations[3].reshape(3)
        center13 = 0.5 * (t1 + t3)
        centers.append(center13)

    # (2,4) 대각선이 모두 보이는 경우
    if has24:
        t2 = tag_translations[2].reshape(3)
        t4 = tag_translations[4].reshape(3)
        center24 = 0.5 * (t2 + t4)
        centers.append(center24)

    # 한 쌍 또는 두 쌍의 중점을 평균 내어 원점으로 사용
    origin_cam = np.mean(centers, axis=0)

    # --- 태그 회전행렬 평균 후 SVD로 정규화 (태그 축 기준 world 회전) -----
    R_list = [tag_rotations[tid] for tid in CORNER_TAG_IDS if tid in present]
    R_stack = np.stack(R_list, axis=0)  # (N, 3, 3)
    R_mean = R_stack.mean(axis=0)       # (3, 3)

    U, _, Vt = np.linalg.svd(R_mean)
    R_cam_world_tag = U @ Vt  # world(tag 축) -> camera

    # det(R) < 0 인 경우 반사(reflection) 방지
    if np.linalg.det(R_cam_world_tag) < 0:
        U[:, -1] *= -1
        R_cam_world_tag = U @ Vt

    # ---------------------------------------------------------
    # 로봇 베이스 축 정의 반영
    #
    # p_cam = R_cam_world_tag * p_world_tag + t
    # p_world_tag = R_ROBOT_TO_TAG * p_world_robot
    # => p_cam = (R_cam_world_tag @ R_ROBOT_TO_TAG) * p_world_robot + t
    # 따라서 R_cam_world_robot = R_cam_world_tag @ R_ROBOT_TO_TAG
    # ---------------------------------------------------------
    R_cam_world_robot = R_cam_world_tag @ R_ROBOT_TO_TAG

    # --- 4x4 변환행렬 구성 (world = 로봇 베이스 프레임) --------------------
    T_camera_world = np.eye(4, dtype=float)
    T_camera_world[0:3, 0:3] = R_cam_world_robot
    T_camera_world[0:3, 3] = origin_cam

    return T_camera_world


def save_extrinsic_to_json(T_camera_world, json_path):
    """
    extrinsic 결과를 JSON 파일로 저장하는 함수.

    저장 내용:
      - T_camera_world : world(=robot base) -> camera (4x4)
      - T_world_camera : camera -> world(=robot base) (4x4)
    """
    T_world_camera = np.linalg.inv(T_camera_world)

    data = {
        "T_camera_world": T_camera_world.tolist(),   # world(=robot base) -> camera
        "T_world_camera": T_world_camera.tolist(),   # camera -> world(=robot base)
        "comment": (
            "World frame is robot base. "
            "Axes mapping: +x_tag->+x_robot, +y_tag->-y_robot, +z_tag->-z_robot. "
            "T_camera_world: world->camera (4x4). "
            "World origin: midpoint(s) of visible diagonal pairs (1,3) and/or (2,4). "
            "Orientation: average of rotations of visible corner tags (1,2,3,4), "
            "then converted from tag axes to robot base axes."
        ),
        "corner_tag_ids_clockwise": CORNER_TAG_IDS,
        "corner_tag_sizes_m": CORNER_TAG_SIZES_M,
        "saved_at": datetime.datetime.now().isoformat(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_extrinsic_image(frame, img_path):
    """
    현재 화면(frame)을 JPEG 이미지로 저장하는 함수.
    """
    cv2.imwrite(img_path, frame)


def draw_bottom_left_lines(frame, lines, color=(0, 0, 255)):
    """
    화면 좌측 하단에 여러 줄 텍스트를 그리는 유틸 함수.
    lines: [str, str, ...]
    color: BGR 색상 (기본 빨간색)
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_height = 30

    # 아래에서 위로 올라가며 그리기
    num = len(lines)
    for i, text in enumerate(reversed(lines)):
        y = h - 20 - i * line_height
        x = 30
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def main():
    # --- 카메라 보정 파라미터 로드 -----------------------------------------
    print("[INFO] 카메라 보정 파라미터(JSON) 불러오는 중...")
    K, dist_coeffs, camera_params = load_camera_params(CALIB_JSON_PATH)
    print("[INFO] K (카메라 내부 행렬):")
    print(K)
    print("[INFO] 왜곡 계수(dist_coeffs):", dist_coeffs)

    # --- AprilTag 검출기 생성 ----------------------------------------------
    detector = create_detector()

    # --- 태그 ID -> 실제 크기 (m) 매핑 생성 -------------------------------
    tag_size_by_id = {
        tag_id: size_m for tag_id, size_m in zip(CORNER_TAG_IDS, CORNER_TAG_SIZES_M)
    }

    # pupil_apriltags Detector.detect() 에 넣을 기준 태그 크기
    # 여기서는 첫 번째 태그 크기를 기준으로 사용
    tag_size_for_detect = float(CORNER_TAG_SIZES_M[0])

    # --- 카메라 열기 -------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. CAMERA 연결 또는 INDEX를 확인하세요.")
        return

    print("[INFO] ESC 키: 종료, SPACE 키: extrinsic 후보 장면 캡처")
    print("[INFO] (1,3) 또는 (2,4) 태그 쌍 중 하나 이상이 보이는 상태에서 SPACE를 눌러야 합니다.")

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    json_path = os.path.join(current_dir, JSON_FILENAME)
    img_path = os.path.join(current_dir, IMG_FILENAME)

    # 상태 메시지(일시적으로 띄울 문구)를 위한 변수
    status_lines = []
    status_frames_left = 0  # 남은 프레임 수 (0이면 표시 안 함)

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
            tag_size=tag_size_for_detect,  # 기준 크기
        )

        # 태그의 중심 픽셀 좌표와 3D 위치(카메라 기준), 회전행렬을 저장할 딕셔너리
        tag_centers_px = {}       # {tag_id: (u,v)}
        tag_translations = {}     # {tag_id: t(3x1)}
        tag_rotations = {}        # {tag_id: R(3x3)}

        # 검출된 태그가 하나도 없으면 안내 출력 (오른쪽 상단)
        if len(detections) == 0:
            text = "No tags detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            h, w = frame.shape[:2]
            x = w - tw - 30
            y = 40
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )
        else:
            # 각 태그에 대해 화면에 표시 및 위치 저장
            for det in detections:
                tag_id = det.tag_id

                # R_raw: 회전 행렬 (3x3)
                R_raw = np.array(det.pose_R)
                # t_raw: 태그 한 변의 길이를 tag_size_for_detect 로 가정했을 때의 번역 (3x1)
                t_raw = np.array(det.pose_t)

                # 이 태그의 "실제" 한 변 길이(미터)
                size_true = tag_size_by_id.get(tag_id, tag_size_for_detect)
                scale = size_true / tag_size_for_detect

                # 실제 크기에 맞게 번역 벡터를 스케일링
                t_scaled = t_raw * scale

                tag_rotations[tag_id] = R_raw
                tag_translations[tag_id] = t_scaled

                # 태그 중심 픽셀 좌표
                c_x, c_y = det.center
                center_px = (int(c_x), int(c_y))
                tag_centers_px[tag_id] = center_px

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

                # --- 2) 태그 중심 위치에 점 찍기 (초록색) -------------------
                cv2.circle(frame, center_px, 4, (0, 255, 0), -1)

                # --- 3) 코너 태그(1,2,3,4)라면 빨간 점(테두리)로 강조 --------
                if tag_id in CORNER_TAG_IDS:
                    cv2.circle(frame, center_px, 6, (0, 0, 255), 2)  # 빨간 원 테두리

                # --- 4) 태그의 로컬 좌표축(X,Y,Z) 그리기 -------------------
                axis_len = 0.04  # 4 cm 정도 길이
                origin_cam = t_scaled.reshape(3)  # 태그 중심(카메라 좌표계)

                x_axis_cam = origin_cam + R_raw[:, 0] * axis_len
                y_axis_cam = origin_cam + R_raw[:, 1] * axis_len
                z_axis_cam = origin_cam + R_raw[:, 2] * axis_len

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
                        cv2.line(frame, o_uv, z_uv, (255, 0, 0), 2)

                # --- 5) 태그 ID 및 (x,y,z) 위치 텍스트로 표시 --------------
                x_m, y_m, z_m = t_scaled.flatten()
                text_id = f"ID: {tag_id}"
                text_pos = f"({x_m:.3f},{y_m:.3f},{z_m:.3f})"

                # ID는 태그 위쪽에
                cv2.putText(
                    frame,
                    text_id,
                    (center_px[0] + 5, center_px[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # 좌표는 태그 아래쪽에 (글씨 작게)
                cv2.putText(
                    frame,
                    text_pos,
                    (center_px[0] + 5, center_px[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # --- (1,3)과 (2,4)를 잇는 빨간 선 및 중점 표시 --------------------
        # (둘 중 보이는 쌍들만 표시)
        # (1,3) 대각선
        if 1 in tag_centers_px and 3 in tag_centers_px:
            p1 = tag_centers_px[1]
            p3 = tag_centers_px[3]
            cv2.line(frame, p1, p3, (0, 0, 255), 2)  # 빨간 선
            mid13 = ((p1[0] + p3[0]) // 2, (p1[1] + p3[1]) // 2)
            cv2.circle(frame, mid13, 5, (0, 0, 255), -1)

        # (2,4) 대각선
        if 2 in tag_centers_px and 4 in tag_centers_px:
            p2 = tag_centers_px[2]
            p4 = tag_centers_px[4]
            cv2.line(frame, p2, p4, (0, 0, 255), 2)  # 빨간 선
            mid24 = ((p2[0] + p4[0]) // 2, (p2[1] + p4[1]) // 2)
            cv2.circle(frame, mid24, 5, (0, 0, 255), -1)

        # --- ESC / SPACE 안내 문구 (검은 글씨, 좌측 상단) -------------------
        cv2.putText(
            frame,
            "ESC: quit, SPACE: capture & save-confirm (Y/N)",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),   # 검은색
            2,
            cv2.LINE_AA,
        )

        # --- 상태 메시지(있으면) 화면 좌측 하단에 빨간 글씨로 표시 ----------
        if status_frames_left > 0 and status_lines:
            draw_bottom_left_lines(frame, status_lines, color=(0, 0, 255))
            status_frames_left -= 1

        # --- 결과 화면 보여주기 --------------------------------------------
        cv2.imshow(WINDOW_NAME, frame)

        # --- 키 입력 처리 ---------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        # ESC -> 종료
        if key == 27:
            break

        # SPACE -> extrinsic 계산 후 저장 여부 확인
        if key == 32:
            present = set(tag_translations.keys()) & set(CORNER_TAG_IDS)
            has13 = (1 in present) and (3 in present)
            has24 = (2 in present) and (4 in present)

            if not (has13 or has24):
                # 화면 좌측 하단에 경고 메시지 잠깐 표시
                status_lines = [
                    "show me the tag: (1,3) pair or (2,4) pair"
                ]
                status_frames_left = 60  # 약 1초 정도 (FPS ~60 가정)
                continue

            try:
                T_camera_world = compute_world_from_tags(
                    tag_translations, tag_rotations
                )
            except ValueError as e:
                status_lines = [str(e)]
                status_frames_left = 60
                continue

            # 현재 화면을 캡처해서 멈춘 뒤, 저장 여부를 물어봄
            freeze_frame = frame.copy()

            # 파일 존재 여부 확인
            json_exists = os.path.exists(json_path)
            img_exists = os.path.exists(img_path)

            if json_exists or img_exists:
                prompt_lines = [
                    "extrinsic_calibration_result already exist",
                    "Do you want to save new file? (Y/N)"
                ]
            else:
                prompt_lines = [
                    "Do you want to save extrinsic_calibration_result? (Y/N) ",
                ]

            while True:
                # 고정된 화면 위에 질문 텍스트를 그려서 보여줌
                prompt_frame = freeze_frame.copy()
                draw_bottom_left_lines(prompt_frame, prompt_lines, color=(255, 255, 255))

                cv2.imshow(WINDOW_NAME, prompt_frame)
                key2 = cv2.waitKey(0) & 0xFF  # 여기서 block

                if key2 in (ord('y'), ord('Y')):
                    # JSON + 이미지 저장 후 종료
                    save_extrinsic_to_json(T_camera_world, json_path)
                    save_extrinsic_image(freeze_frame, img_path)
                    # 저장 완료 메시지도 화면에 잠깐 보여주고 종료
                    done_frame = freeze_frame.copy()
                    draw_bottom_left_lines(
                        done_frame,
                        ["Saved. Close the window."],
                        color=(0, 0, 255),
                    )
                    cv2.imshow(WINDOW_NAME, done_frame)
                    cv2.waitKey(500)  # 0.5초 정도 보여주기
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                elif key2 in (ord('n'), ord('N')):
                    # 저장 취소, 실시간 화면으로 복귀
                    status_lines = ["Save cancelled"]
                    status_frames_left = 60
                    break

                elif key2 == 27:  # ESC
                    status_lines = ["ESC pressed: Save cancelled."]
                    status_frames_left = 60
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
