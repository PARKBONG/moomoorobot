import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

def main():
    # RealSense 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Depth와 Color 정렬(align)
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)

    # 카메라 intrinsic 가져오기 (color 기준)
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.ppx,
        intrinsics.ppy,
    )

    # Open3D 시각화 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RealSense RGB PointCloud", width=intrinsics.width, height=intrinsics.height)

    pcd = o3d.geometry.PointCloud()
    is_pcd_added = False

    try:
        while True:
            # 프레임 획득 및 정렬
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Open3D용 이미지 래핑
            o3d_depth = o3d.geometry.Image(depth_image)
            o3d_color = o3d.geometry.Image(color_image)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            # print("depth scale:", depth_scale)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale = 1.0 / depth_scale,  # <-- 이게 핵심
                # depth_trunc = 5.0,
                convert_rgb_to_intensity=False
            )

            # 카메라 좌표계 기준 포인트클라우드 생성
            pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                pinhole_camera_intrinsic
            )

            # Open3D 좌표계 맞추기 (뒤집어서 보기 좋게)
            pcd_new.transform([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])

            # 기존 pcd 객체 업데이트
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors

            if not is_pcd_added:
                vis.add_geometry(pcd)
                is_pcd_added = True

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # 원하면 RGB 이미지도 참고용으로 띄우기 (선택 사항)
            cv2.imshow("Color", color_image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
