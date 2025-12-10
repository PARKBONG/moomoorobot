import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import open3d as o3d


def main():
    # -----------------------
    # YOLO 세그멘테이션 모델 로드
    # -----------------------
    # 이미 다운로드 되어 있다면 그대로 사용, 없으면 자동 다운로드됨
    model = YOLO("yolov8n-seg.pt")
    # model = YOLO("yolo11n-seg.pt")  # 원하면 이렇게 교체

    # -----------------------
    # RealSense D405 파이프라인 설정
    # -----------------------
    pipeline = rs.pipeline()
    config = rs.config()
    # RGB + Depth 둘 다 사용
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("[INFO] Starting RealSense pipeline...")
    profile = pipeline.start(config)
    print("[INFO] Pipeline started. Press ESC to exit.")

    # depth scale (raw depth → meter 변환)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("[INFO] Depth scale:", depth_scale)

    # color intrinsics (픽셀 → 3D 변환용)
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    # -----------------------
    # Open3D Visualizer 초기화
    # -----------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window("Masked PointCloud (Top-1 Seg)", width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    added = False

    # 2D 디스플레이용 OpenCV 창
    win_name = "D405 + YOLO Seg (RGB)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 720)

    depth_trunc_m = 0.7  # D405는 근거리라 0.5~0.7m 정도만 사용

    try:
        while True:
            # -----------------------
            # 프레임 획득 및 Color 기준 정렬
            # -----------------------
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())      # uint16
            color_image = np.asanyarray(color_frame.get_data())      # uint8, BGR

            H, W = depth_image.shape

            # -----------------------
            # YOLO 세그멘테이션 추론 (RGB 기준)
            # -----------------------
            results = model(color_image, verbose=False)
            res = results[0]

            annotated = res.plot()  # 2D 시각화용 (전체 인스턴스)

            # -----------------------
            # 세그멘테이션 중 top-1 인스턴스 선택
            # -----------------------
            mask_bool = None

            if res.masks is not None and res.boxes is not None and len(res.masks.data) > 0:
                # confidence 가장 높은 인스턴스 선택
                confs = res.boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))

                mask_tensor = res.masks.data[best_idx]  # (h, w)
                mask = mask_tensor.cpu().numpy()

                # mask 크기가 입력 이미지와 다를 수 있으므로 resize
                mh, mw = mask.shape
                if (mh, mw) != (H, W):
                    mask = cv2.resize(mask, (W, H))

                # threshold 적용하여 boolean mask 생성
                mask_bool = mask > 0.5

            # -----------------------
            # PointCloud 생성 (mask 적용)
            # -----------------------
            # depth → meter
            depth_m = depth_image.astype(np.float32) * depth_scale

            # 기본 유효 depth 마스크
            valid_depth = (depth_m > 0) & (depth_m < depth_trunc_m)

            if mask_bool is not None:
                combined_mask = valid_depth & mask_bool
            else:
                # 세그멘테이션이 없으면 전체 유효 depth 사용
                combined_mask = valid_depth

            if not np.any(combined_mask):
                # 유효 포인트가 없으면 그냥 스킵
                cv2.imshow(win_name, annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            ys, xs = np.where(combined_mask)  # 이미지 좌표 (row=y, col=x)
            zs = depth_m[ys, xs]

            # 픽셀 → 카메라 좌표계 (X,Y,Z) [m]
            # u = x, v = y
            xs_f = xs.astype(np.float32)
            ys_f = ys.astype(np.float32)

            X = (xs_f - cx) * zs / fx
            Y = (ys_f - cy) * zs / fy
            Z = zs

            points = np.stack((X, Y, Z), axis=-1)  # (N, 3)

            # 색 정보 (BGR → RGB, 0~1)
            colors_bgr = color_image[ys, xs, :]       # (N, 3)
            colors_rgb = colors_bgr[:, ::-1] / 255.0  # BGR→RGB

            # Open3D PointCloud 업데이트
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

            # 보기 편하게 좌표계 뒤집기 (Open3D convention)
            T = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]], dtype=np.float64)
            pcd.transform(T)

            if not added:
                vis.add_geometry(pcd)
                added = True

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # 2D 세그멘테이션 결과 디스플레이
            cv2.imshow(win_name, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC pressed. Exiting...")
                break

    finally:
        print("[INFO] Stopping pipeline and closing windows...")
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
