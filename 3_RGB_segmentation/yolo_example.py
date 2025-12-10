# pip install ultralytics

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def main():
    # -----------------------
    # YOLO 세그멘테이션 모델 로드
    # -----------------------
    # 경량 세그멘테이션 모델 (필요시 "yolo11n-seg.pt"로 변경 가능)
    model = YOLO("yolov8n-seg.pt") # 다른 모델도 쓸수 있음. 여기 보셈; https://docs.ultralytics.com/ko/models/
    # 보통 정확도 올라가면 속도 느려짐. 지금은 낮은정확도, 빠른속력. 
    # model = YOLO("yolo11n-seg.pt")

    # -----------------------
    # RealSense D405 파이프라인 설정 (RGB만 사용)
    # -----------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # D405의 color 스트림 설정 (해상도/프레임은 필요에 따라 변경 가능)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("[INFO] Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("[ERROR] pipeline.start 실패:", e)
        return

    print("[INFO] Pipeline started. Press ESC to exit.")

    window_name = "D405 + YOLO Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            # -----------------------
            # RGB 프레임 획득
            # -----------------------
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                # 프레임 못 받았으면 다음 루프
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # -----------------------
            # YOLO 세그멘테이션 추론
            # -----------------------
            # (원한다면 size=512 등으로 입력 해상도 줄일 수 있음)
            results = model(color_image, verbose=False)

            # 첫 번째 결과에 대한 어노테이션 이미지 생성
            annotated = results[0].plot()  # box + mask + label 오버레이된 이미지

            # -----------------------
            # 화면 표시
            # -----------------------
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[INFO] ESC pressed, exiting...")
                break

    finally:
        print("[INFO] Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()