# pip install pybullet pyvirtualcam opencv-python numpy
# https://github.com/schellingb/UnityCapture
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import pyvirtualcam


def My_Pybullet_CAM_ENV(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
):
    """
    PyBullet를 띄우고, 시뮬레이션 카메라 영상을
    Windows 가상 카메라(pyvirtualcam)로 스트리밍하는 환경.

    기존 코드(실카메라용)는 전혀 수정하지 않고,
    OS 레벨에서 'Virtual Camera'를 새 카메라로 인식하게 만드는 방식입니다.
    """

    # ------------------------------------------------------------------
    # 1) PyBullet 초기화 (GUI 모드: 실제로 장면이 보임)
    # ------------------------------------------------------------------
    physics_client = p.connect(p.GUI)  # 필요시 p.DIRECT 로 변경 가능
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 바닥 및 예제 물체 로드 (원하면 다른 URDF로 교체 가능)
    plane_id = p.loadURDF("plane.urdf")
    cube_start_pos = [0, 0, 0.1]
    cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    cube_id = p.loadURDF("r2d2.urdf", cube_start_pos, cube_start_orn)

    # ------------------------------------------------------------------
    # 2) 카메라 파라미터 설정
    # ------------------------------------------------------------------
    # 시뮬레이션 카메라는 D435 1280x720 환경과 유사한 구성을 가정
    fov = 69.0  # D435 수평 시야각과 비슷하게
    aspect = width / float(height)
    near = 0.01
    far = 5.0

    # 카메라 위치/자세 (원하면 애니메이션 가능)
    cam_target_pos = [0, 0, 0.5]
    cam_distance = 1.0
    cam_yaw = 45.0
    cam_pitch = -30.0
    cam_roll = 0.0
    up_axis_index = 2  # z-up

    # ------------------------------------------------------------------
    # 3) 가상 카메라 오픈 (pyvirtualcam)
    # ------------------------------------------------------------------
    with pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=fps,
        fmt=pyvirtualcam.PixelFormat.BGR,
        print_fps=True,
    ) as cam:
        print(f"[INFO] Virtual camera started: {cam.device}")

        try:
            # 메인 루프
            dt = 1.0 / fps
            last_time = time.time()

            while True:
                # PyBullet 시뮬레이션 스텝
                p.stepSimulation()

                # 카메라 뷰/프로젝션 행렬 계산
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=cam_target_pos,
                    distance=cam_distance,
                    yaw=cam_yaw,
                    pitch=cam_pitch,
                    roll=cam_roll,
                    upAxisIndex=up_axis_index,
                )

                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=fov,
                    aspect=aspect,
                    nearVal=near,
                    farVal=far,
                )

                # 카메라 이미지 렌더링
                img = p.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                )

                # PyBullet가 int32 배열을 주는 경우가 있으므로 먼저 uint8로 변환
                rgba = np.reshape(img[2], (height, width, 4)).astype(np.uint8)
                rgb  = rgba[:, :, :3]

                # 이제 cvtColor에 넣어도 안전
                frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # 가상 카메라로 전송
                cam.send(frame_bgr)
                cam.sleep_until_next_frame()

                # FPS 맞추기용 (필요시 생략 가능)
                now = time.time()
                elapsed = now - last_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                last_time = now

        except KeyboardInterrupt:
            print("\n[INFO] My_Pybullet_CAM_ENV() stopped by user.")

        finally:
            p.disconnect(physics_client)


if __name__ == "__main__":
    # 이 파일을 직접 실행하면 곧바로 PyBullet + 가상 카메라 환경이 켜집니다.
    My_Pybullet_CAM_ENV()
