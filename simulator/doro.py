import os
import math
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import pyvirtualcam


# ==========================
# 전역 설정
# ==========================

# 카메라 해상도 / FPS
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 60  # 목표 FPS

# 도로/환경 파라미터
ROAD_LENGTH = 20.0        # 도로 전체 길이(m) 대략
NUM_POINTS = 80           # 곡선을 구성할 샘플 포인트 수
ROAD_HALF_WIDTH = 0.7     # 도로 폭의 절반 (전체 폭 ≈ 1.4m)
ROAD_THICKNESS = 0.02

CENTER_LINE_WIDTH = 0.05
SIDE_LINE_WIDTH = 0.05

OBSTACLE_SIZE = [0.3, 0.2, 0.2]  # [x_half, y_half, z_half]
NUM_OBSTACLES = 5

# 제어 키 (OpenCV 창 기준, WASD + ESC)
KEY_ESC = 27
KEY_W = ord('w')
KEY_S = ord('s')
KEY_A = ord('a')
KEY_D = ord('d')


# ==========================
# 도로/장애물 생성
# ==========================

def generate_road_points():
    """
    x를 0~ROAD_LENGTH 사이에서 균등분할하고,
    y를 약간 구불구불한 곡선(sin 기반)으로 생성.
    """
    xs = np.linspace(0.0, ROAD_LENGTH, NUM_POINTS)
    ys = 0.6 * np.sin(xs * 0.4)   # 적당히 휘는 곡선
    points = np.stack([xs, ys], axis=1)
    return points


def create_ground():
    """체크무늬 plane 대신 단색 박스로 바닥 생성."""
    ground_half_extents = [50.0, 50.0, 0.05]  # 충분히 큰 바닥
    ground_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=ground_half_extents,
    )
    ground_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=ground_half_extents,
        rgbaColor=[0.8, 0.8, 0.8, 1.0],  # 연한 회색 단색
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=ground_col,
        baseVisualShapeIndex=ground_vis,
        basePosition=[ROAD_LENGTH / 2.0, 0.0, -ground_half_extents[2]],
    )


def create_road(points):
    """
    곡선을 따라 도로 패치(회색 바닥) + 중앙선(검정) + 양측 빨간 경계선(box)들을 생성.
    각 선분마다 하나의 road patch를 만든다고 생각하면 됨.
    """
    road_ids = []
    center_line_ids = []
    side_line_ids = []

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        seg = p1 - p0
        length = np.linalg.norm(seg)
        if length < 1e-6:
            continue

        # 선분 중앙 좌표
        cx, cy = (p0 + p1) / 2.0
        cz = 0.0

        # yaw 계산 (z축 기준 회전)
        yaw = math.atan2(seg[1], seg[0])
        orn = p.getQuaternionFromEuler([0, 0, yaw])

        # 도로 본체(회색)
        road_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length / 2.0, ROAD_HALF_WIDTH, ROAD_THICKNESS / 2.0]
        )
        road_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2.0, ROAD_HALF_WIDTH, ROAD_THICKNESS / 2.0],
            rgbaColor=[0.4, 0.4, 0.4, 1.0]
        )
        road_id = p.createMultiBody(
            baseCollisionShapeIndex=road_col,
            baseVisualShapeIndex=road_vis,
            basePosition=[cx, cy, cz - ROAD_THICKNESS / 2.0],
            baseOrientation=orn
        )
        road_ids.append(road_id)

        # 중앙선(검정 얇은 박스)
        center_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2.0, CENTER_LINE_WIDTH / 2.0, ROAD_THICKNESS / 4.0],
            rgbaColor=[0.0, 0.0, 0.0, 1.0]
        )
        center_id = p.createMultiBody(
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=center_vis,
            basePosition=[cx, cy, cz + 1e-3],
            baseOrientation=orn
        )
        center_line_ids.append(center_id)

        # 도로 좌우 방향의 법선 벡터 (z축 기준)
        dx, dy = seg / length
        nx, ny = -dy, dx  # 좌측 방향

        offset = ROAD_HALF_WIDTH
        # 왼쪽 빨간 경계선
        sx = cx + nx * offset
        sy = cy + ny * offset
        side_vis_l = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2.0, SIDE_LINE_WIDTH / 2.0, ROAD_THICKNESS / 4.0],
            rgbaColor=[1.0, 0.0, 0.0, 1.0]
        )
        side_id_l = p.createMultiBody(
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=side_vis_l,
            basePosition=[sx, sy, cz + 1e-3],
            baseOrientation=orn
        )
        side_line_ids.append(side_id_l)

        # 오른쪽 빨간 경계선
        sx = cx - nx * offset
        sy = cy - ny * offset
        side_vis_r = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2.0, SIDE_LINE_WIDTH / 2.0, ROAD_THICKNESS / 4.0],
            rgbaColor=[1.0, 0.0, 0.0, 1.0]
        )
        side_id_r = p.createMultiBody(
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=side_vis_r,
            basePosition=[sx, sy, cz + 1e-3],
            baseOrientation=orn
        )
        side_line_ids.append(side_id_r)

    return {
        "roads": road_ids,
        "center_lines": center_line_ids,
        "side_lines": side_line_ids,
    }


def create_obstacles_with_tags(points, tag_texture_paths):
    """
    도로 상의 몇 점을 골라 사각형 장애물을 올리고,
    각 장애물의 네 모서리 근처에 AprilTag 텍스처를 붙인 얇은 판을 배치.
    tag_texture_paths: [tag0.png, tag1.png, ...]
    """
    obstacle_ids = []
    tag_plane_ids = []

    # 장애물을 놓을 인덱스들을 간격을 두고 선택
    idx_candidates = np.linspace(10, len(points) - 10, NUM_OBSTACLES, dtype=int)

    for idx_i, idx in enumerate(idx_candidates):
        center_xy = points[idx]
        x, y = float(center_xy[0]), float(center_xy[1])
        z = OBSTACLE_SIZE[2]  # half z

        # 장애물 본체 (회색 박스)
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=OBSTACLE_SIZE
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=OBSTACLE_SIZE,
            rgbaColor=[0.7, 0.7, 0.7, 1.0]
        )
        obs_id = p.createMultiBody(
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z]
        )
        obstacle_ids.append(obs_id)

        # 태그 텍스처 로드 (없으면 그냥 흰 판)
        tag_tex_path = tag_texture_paths[idx_i % len(tag_texture_paths)]
        if os.path.exists(tag_tex_path):
            tex_id = p.loadTexture(tag_tex_path)
        else:
            tex_id = None

        # 모서리 네 점 (top face corners) 근처에 얇은 판 생성
        hx, hy, hz = OBSTACLE_SIZE
        corners_local = [
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
            [-hx, -hy, +hz],
            [+hx, -hy, +hz],
        ]

        plane_half = [0.06, 0.001, 0.06]  # 작은 정사각형 판 (태그용)

        # 태그 판용 visual shape (텍스처는 나중에 changeVisualShape로 입힘)
        plane_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=plane_half,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )

        for c in corners_local:
            px = x + c[0]
            py = y + c[1]
            pz = z + c[2] + plane_half[1]

            plane_id = p.createMultiBody(
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=plane_vis,
                basePosition=[px, py, pz],
                baseOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            )

            # 여기서 텍스처 적용 (버전 호환용 try/except)
            if tex_id is not None:
                try:
                    p.changeVisualShape(plane_id, -1, textureUniqueId=tex_id)
                except TypeError:
                    # 어떤 버전에서는 textureUniqueId 인자가 없을 수도 있으므로, 그 경우는 그냥 흰 판으로 둠
                    pass

            tag_plane_ids.append(plane_id)

    return {
        "obstacles": obstacle_ids,
        "tags": tag_plane_ids,
    }


# ==========================
# 자동차 / 카메라 관련
# ==========================

def spawn_little_car():
    """
    PyBullet 내장 racecar를 '꼬마 자동차'로 사용.
    """
    car_urdf = os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf")
    start_pos = [0.0, 0.0, 0.2]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    car_id = p.loadURDF(car_urdf, start_pos, start_orn)
    return car_id


def setup_racecar_joints(car_id):
    """
    racecar.urdf 내에서 steering / wheel 조인트를 자동으로 찾아서 반환.
    이름에 'steer' 또는 'wheel' 이 포함된 조인트를 사용합니다.
    """
    num_joints = p.getNumJoints(car_id)
    steering_joints = []
    wheel_joints = []

    print("[INFO] Racecar joints:")
    for j in range(num_joints):
        info = p.getJointInfo(car_id, j)
        name = info[1].decode("utf-8")
        print(f"  idx={j}, name={name}")
        lname = name.lower()
        if "steer" in lname:
            steering_joints.append(j)
        if "wheel" in lname:
            wheel_joints.append(j)

    print("[INFO] steering_joints:", steering_joints)
    print("[INFO] wheel_joints   :", wheel_joints)

    # 모터를 모두 '힘 없음'으로 초기화해서, 우리가 직접 joint position을 설정해도
    # 이상한 토크가 붙지 않도록 함
    for j in steering_joints + wheel_joints:
        p.setJointMotorControl2(
            car_id, j,
            p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
        )

    return steering_joints, wheel_joints


def compute_ego_camera(car_id):
    """
    자동차에 붙어있는 카메라의 위치와 방향을 계산.
    간단히 base 위치 + 차량 전방 방향 기준으로 오프셋.
    """
    base_pos, base_orn = p.getBasePositionAndOrientation(car_id)
    base_pos = np.array(base_pos)

    # Euler로 변환
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])

    # 카메라 오프셋 (차체 앞쪽 + 위쪽)
    cam_offset = 0.3 * forward + 0.2 * up
    cam_eye = base_pos + cam_offset
    cam_target = cam_eye + forward * 2.0

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_eye.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=up.tolist()
    )

    return view_matrix


def set_topdown_debug_camera(points):
    """
    PyBullet GUI용 카메라를 위에서 내려다보는 "맵 뷰"로 설정.
    도로 중앙 근처를 가운데로 잡음.
    """
    center = points[len(points) // 2]
    cx, cy = float(center[0]), float(center[1])
    target_pos = [cx, cy, 0.0]

    distance = 10.0
    yaw = 90.0
    pitch = -89.0  # 거의 수직
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target_pos
    )


# ==========================
# 메인 환경
# ==========================

def My_Pybullet_CAM_ENV():
    """
    PyBullet GUI에서:
      - 구불구불한 도로 + 검은 중앙선 + 빨간 경계선
      - 장애물 + 모서리 AprilTag
      - 꼬마 자동차(racecar)
      - GUI 카메라는 맵 뷰(탑다운)
    를 보여주고,
    동시에 자동차에 달린 카메라 화면을 OBS Virtual Camera와
    OpenCV 창으로 스트리밍하며, OpenCV 창에서 WASD로 조종.
    """

    # 1) PyBullet 연결 및 기본 세팅
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # 바닥 플레인 대신 단색 바닥
    create_ground()

    # 2) 도로 곡선 생성 및 도로/라인 만들기
    road_points = generate_road_points()
    _road_env = create_road(road_points)

    # 3) 장애물 + AprilTag 텍스처
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tag_dir = os.path.join(script_dir, "..", "data", "apriltags")
    tag_texture_paths = [
        os.path.join(tag_dir, "tag36h11_0.png"),
        os.path.join(tag_dir, "tag36h11_1.png"),
        os.path.join(tag_dir, "tag36h11_2.png"),
        os.path.join(tag_dir, "tag36h11_3.png"),
    ]
    _obstacle_env = create_obstacles_with_tags(road_points, tag_texture_paths)

    # 4) 꼬마 자동차 스폰
    car_id = spawn_little_car()

    # 4-1) 자동차 조인트 인덱스 (조향/바퀴)
    steering_joints, wheel_joints = setup_racecar_joints(car_id)

    # 4-2) 자동차 상태 변수 (kinematic)
    steering_angle = 0.0      # 조향각 [rad]
    car_speed = 0.0           # 속도 [m/s]
    max_steering_angle = 0.6  # ±34도 정도
    max_speed = 5.0           # m/s
    accel = 0.1               # 전/후진 시 가속량
    brake_decay = 0.08        # 입력 없을 때 감속량
    steer_step = 0.03         # 조향키 1번당 변화량

    wheel_radius = 0.05       # 바퀴 반지름 (대략)
    wheel_rotation = 0.0      # 바퀴 회전각 누적
    wheelbase = 0.6           # 앞/뒤 축 거리 (대략)

    # 5) GUI 카메라를 탑다운 맵 뷰로 설정
    set_topdown_debug_camera(road_points)

    # FPS 모니터링용
    ema_fps = None
    frame_count = 0

    # 6) 가상 카메라(OBS Virtual Camera) 열기
    with pyvirtualcam.Camera(
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        fps=CAM_FPS,
        fmt=pyvirtualcam.PixelFormat.BGR,
        backend="obs",
        print_fps=True,
    ) as cam:
        print("[INFO] OBS Virtual Camera device:", cam.device)
        print("[INFO] Streaming ego-camera view to OBS Virtual Camera.")
        print("[INFO] Control keys in OpenCV window: W/S (forward/back), A/D (steer), ESC (quit).")

        try:
            while True:
                loop_start = time.time()

                # -----------------------------
                # (A) 차량 제어: OpenCV 키 입력
                # -----------------------------
                key = cv2.waitKey(1) & 0xFF

                if key == KEY_ESC:
                    print("[INFO] ESC pressed. Exiting simulation loop.")
                    break

                # 전진/후진
                if key == KEY_W:
                    car_speed += accel
                elif key == KEY_S:
                    car_speed -= accel

                # 조향
                if key == KEY_A:
                    steering_angle += steer_step
                elif key == KEY_D:
                    steering_angle -= steer_step

                # 속도/조향 제한
                car_speed = max(-max_speed, min(max_speed, car_speed))
                steering_angle = max(-max_steering_angle, min(max_steering_angle, steering_angle))

                # 입력 없을 때 서서히 감속
                if key not in (KEY_W, KEY_S):
                    if abs(car_speed) < brake_decay:
                        car_speed = 0.0
                    else:
                        car_speed -= brake_decay * np.sign(car_speed)

                # -----------------------------
                # (B) Kinematic 업데이트
                # -----------------------------
                base_pos, base_orn = p.getBasePositionAndOrientation(car_id)
                roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

                forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])

                v = car_speed  # [m/s]
                lin_vel = (forward * v).tolist()

                if abs(steering_angle) > 1e-4:
                    yaw_rate = v * math.tan(steering_angle) / wheelbase
                else:
                    yaw_rate = 0.0
                ang_vel = [0.0, 0.0, yaw_rate]

                p.resetBaseVelocity(car_id, linearVelocity=lin_vel, angularVelocity=ang_vel)

                # -----------------------------
                # (C) 바퀴 / 조향 joint 시각화
                # -----------------------------
                for j in steering_joints:
                    p.setJointMotorControl2(
                        car_id, j,
                        p.POSITION_CONTROL,
                        targetPosition=steering_angle,
                        force=1.0,
                    )

                wheel_rotation += (v / max(wheel_radius, 1e-3)) * (1.0 / max(CAM_FPS, 1))
                for j in wheel_joints:
                    p.setJointMotorControl2(
                        car_id, j,
                        p.POSITION_CONTROL,
                        targetPosition=wheel_rotation,
                        force=1.0,
                    )

                # -----------------------------
                # (D) 물리 시뮬레이션 스텝
                # -----------------------------
                p.stepSimulation()

                # -----------------------------
                # (E) 자동차 카메라 렌더링
                # -----------------------------
                view_matrix = compute_ego_camera(car_id)

                fov = 69.0
                aspect = CAM_WIDTH / float(CAM_HEIGHT)
                near = 0.01
                far = 50.0

                # 빠른 렌더링을 위한 flags (segmentation mask 비활성화)
                flags = p.ER_NO_SEGMENTATION_MASK

                img = p.getCameraImage(
                    CAM_WIDTH,
                    CAM_HEIGHT,
                    viewMatrix=view_matrix,
                    projectionMatrix=p.computeProjectionMatrixFOV(
                        fov=fov,
                        aspect=aspect,
                        nearVal=near,
                        farVal=far
                    ),
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    flags=flags,
                )

                rgba = np.reshape(img[2], (CAM_HEIGHT, CAM_WIDTH, 4)).astype(np.uint8)
                rgb = rgba[:, :, :3]
                frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # -----------------------------
                # (F) OBS 가상카메라 + OpenCV 창으로 동시에 송신
                # -----------------------------
                cam.send(frame_bgr)
                cam.sleep_until_next_frame()

                cv2.imshow("Sim Ego Camera (Control Window)", frame_bgr)

                # -----------------------------
                # (G) FPS 측정 (간단한 EMA)
                # -----------------------------
                frame_time = time.time() - loop_start
                if frame_time > 0:
                    inst_fps = 1.0 / frame_time
                    if ema_fps is None:
                        ema_fps = inst_fps
                    else:
                        alpha = 0.1
                        ema_fps = alpha * inst_fps + (1 - alpha) * ema_fps

                    frame_count += 1
                    if frame_count % 120 == 0:
                        print(f"[INFO] Approx. loop FPS: {ema_fps:.1f}")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (KeyboardInterrupt).")

        finally:
            cv2.destroyAllWindows()
            p.disconnect(physics_client)


if __name__ == "__main__":
    My_Pybullet_CAM_ENV()
