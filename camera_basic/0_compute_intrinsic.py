import os
import cv2
import numpy as np
import datetime
import json

# Checker Board Download:
# https://markhedleyjones.com/projects/calibration-checkerboard-collection

# ============================================================
# 설정 부분 (사용자가 바꿔도 되는 값)
# ============================================================

# 1) 체커보드 내부 코너 개수
#    예) PDF에 "8 x 6 inner corners" 라고 쓰여 있으면 (8, 6)
#        (가로 코너 8개, 세로 코너 6개)
# 감지가 안되면 PATTERN_SIZE 문제일 가능성이 큼.
PATTERN_SIZE = (8, 6)  # (가로 코너 개수, 세로 코너 개수)

# 2) 한 칸의 실제 크기 (미터 단위)
#    예) 3 cm 체커보드이면 0.03
# 프린트 후 자로 직접 잴것
SQUARE_SIZE = 0.03  # 3 cm

# 3) 카메라 인덱스
CAMERA_INDEX = 1 # Intel Realsense 기준으로 카메라가 2개 연결된걸로 인식됨. 0혹은 1시도 필요. 

# 4) 최소 촬영 장수 (너무 적으면 보정 정확도가 떨어짐)
MIN_SAMPLES = 50

# 5) 시각화용 윈도우 크기 조정 비율 (기본값: 1.0)
ADJUST_WIN_SIZE = 1.0

# 6) 샘플 이미지 저장 여부
SAVE = False  # True면 images/0.jpeg, 1.jpeg ... 저장, False면 저장 안 함

# ➜ 추가: 캡처 해상도 (width, height). None이면 기본 해상도 사용
PIXEL_SIZE = (1080, 1080)

# 설정 부분 끝 =================================================

# ============================================================
# 체커보드 한 장에 대한 3D 점 좌표 준비
# ============================================================
def create_object_points(PATTERN_SIZE, SQUARE_SIZE):
    """
    체커보드가 평면(z=0)에 놓여 있다고 가정하고,
    모든 코너의 3D 좌표(월드 좌표)를 만드는 함수.

    PATTERN_SIZE: (가로 코너 개수, 세로 코너 개수)
    SQUARE_SIZE : 한 칸의 실제 길이 [m]
    """
    nx, ny = PATTERN_SIZE  # nx: 가로, ny: 세로

    # 총 코너 수: nx * ny
    objp = np.zeros((ny * nx, 3), np.float32)
    # (0,0), (1,0), ..., (nx-1, ny-1) 형태의 그리드를 만들어서
    # square_size를 곱해 실제 길이로 변경
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    return objp


# ============================================================
# 카메라에서 이미지 캡처 + 체커보드 코너 찾기
# ============================================================
def capture_calibration_images(cap, PATTERN_SIZE, objp, MIN_SAMPLES, SAVE):
    """
    실시간으로 카메라 화면을 보여주면서,
    체커보드를 인식하면 SPACE로 한 장씩 캡처하는 함수.

    ESC를 누르면 캡처를 종료하고,
    모인 objpoints, imgpoints를 돌려준다.
    """
    objpoints = []  # 3D 점 (월드 좌표계)
    imgpoints = []  # 2D 점 (이미지 좌표계)
    sample_count = 0

    # 이미지 저장 폴더 준비
    images_dir = None
    if SAVE:
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        images_dir = os.path.join(current_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"[INFO] 이미지 저장 모드: ON  (폴더: {images_dir})")
    else:
        print("[INFO] 이미지 저장 모드: OFF")

    print("=== 카메라 보정 시작 ===")
    print(f"- 체커보드 코너 개수 (가로, 세로): {PATTERN_SIZE}")
    print(f"- 한 칸 크기: {SQUARE_SIZE} m")
    print("")
    print("사용 방법:")
    print("  1) 체커보드를 화면에 비추세요.")
    print("  2) 체커보드가 인식되면, SPACE 키를 눌러 한 장 저장합니다.")
    print("  3) 여러 각도/거리에서 최소 {0}장 이상 저장해주세요.".format(MIN_SAMPLES))
    print("  4) 충분히 모였다고 생각되면 ESC 키를 눌러 보정을 진행합니다.")
    print("")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 읽지 못했습니다.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

        display = frame.copy()

        if found:
            # 코너를 더 정확하게 찾기 위한 서브픽셀 보정
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria,
            )

            cv2.drawChessboardCorners(display, PATTERN_SIZE, corners_refined, found)

            msg = "Chessboard detected: SPACE = capture, ESC = finish"
            cv2.putText(display, msg, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        cv2.LINE_AA)
        else:
            msg = "Show the chessboard clearly. (ESC = quit)"
            cv2.putText(display, msg, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        cv2.LINE_AA)

        # 현재까지 저장된 샘플 수 표시
        cv2.putText(display, f"Samples: {sample_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # 윈도우 크기 조정 (시각화용)
        if ADJUST_WIN_SIZE is not None and ADJUST_WIN_SIZE > 0:
            h, w = display.shape[:2]
            new_w = int(w * ADJUST_WIN_SIZE)
            new_h = int(h * ADJUST_WIN_SIZE)
            display_resized = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_resized = display
            
        cv2.imshow("Camera Calibration", display_resized)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            # 촬영 종료
            break
        elif key == 32 and found:  # SPACE + 체커보드 인식
            # 저장 전 sample index (파일 이름용)
            sample_idx = sample_count

            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            sample_count += 1
            print(f"[INFO] 캡처 완료: {sample_count} 장")

            # 이미지 파일 저장
            if SAVE and images_dir is not None:
                img_path = os.path.join(images_dir, f"{sample_idx}.jpeg")
                cv2.imwrite(img_path, frame)
                print(f"        → 이미지 저장: {img_path}")

            if sample_count >= MIN_SAMPLES:
                print("충분한 샘플이 모였습니다. 원하시면 ESC로 종료 후 보정을 진행하세요.")

    return objpoints, imgpoints, gray if 'gray' in locals() else None


# ============================================================
# 카메라 보정(calibrateCamera) 실행
# ============================================================
def run_calibration(objpoints, imgpoints, image_size):
    """
    OpenCV의 calibrateCamera 함수를 이용해
    카메라 행렬(K)과 왜곡 계수(dist_coeffs)를 추정.
    """
    print("\n=== 카메라 보정(calibrateCamera) 실행 중... ===")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    print("\n=== 보정 결과 ===")
    print(f"재투영 RMS 오차 (작을수록 좋음): {ret:.6f}\n")
    print("카메라 행렬 K (3x3):")
    print(camera_matrix)
    print("\n왜곡 계수 [k1, k2, p1, p2, k3, ...]:")
    print(dist_coeffs.ravel())

    return ret, camera_matrix, dist_coeffs


# ============================================================
# JSON으로 결과 저장 (cx, cy, fx, fy 형태로)
# ============================================================
def save_to_json(camera_matrix, dist_coeffs, image_size,
                 PATTERN_SIZE, SQUARE_SIZE, reprojection_error):
    """
    보정 결과를 JSON 파일로 저장.
    카메라 행렬은 3x3 전체 대신 fx, fy, cx, cy로 저장.
    """
    # K에서 fx, fy, cx, cy 추출
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"intrinsic_calibration_result"
    json_filename = f"{base_filename}.json"
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    json_filename = os.path.join(current_dir, json_filename)

    json_data = {
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "image_size": list(image_size),             # [width, height]
        "PATTERN_SIZE": list(PATTERN_SIZE),         # [nx, ny]
        "SQUARE_SIZE": float(SQUARE_SIZE),          # [m]
        "reprojection_error": float(reprojection_error),
    }

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ JSON 파일로 저장 완료: {json_filename}")
    print("\n나중에 불러오는 예시:")
    print("  import json")
    print(f"  with open('{json_filename}', 'r', encoding='utf-8') as f:")
    print("      data = json.load(f)")
    print("  fx = data['intrinsics']['fx']")
    print("  fy = data['intrinsics']['fy']")
    print("  cx = data['intrinsics']['cx']")
    print("  cy = data['intrinsics']['cy']")


# ============================================================
# 메인 함수
# ============================================================
def main():
    # 체커보드 한 장에 대한 3D 점 좌표 생성
    objp = create_object_points(PATTERN_SIZE, SQUARE_SIZE)

    # 카메라 열기
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(인덱스 {CAMERA_INDEX})를 열 수 없습니다.")

    # ➜ 추가: 원하는 해상도 요청 및 실제 해상도 확인
    if PIXEL_SIZE is not None:
        req_w, req_h = PIXEL_SIZE
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 요청 해상도: {req_w} x {req_h}")
        print(f"[INFO] 실제 카메라 해상도: {actual_w} x {actual_h}")
    else:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 기본 카메라 해상도 사용: {actual_w} x {actual_h}")

    # 체커보드 캡처
    objpoints, imgpoints, gray = capture_calibration_images(
        cap, PATTERN_SIZE, objp, MIN_SAMPLES, SAVE
    )

    cap.release()
    cv2.destroyAllWindows()

    # 샘플 개수 확인
    if len(objpoints) < MIN_SAMPLES:
        print(f"\n[경고] 보정에 필요한 최소 샘플 수({MIN_SAMPLES})보다 적습니다.")
        print(f"현재 샘플 수: {len(objpoints)}")
        print("정확한 결과를 위해, 다시 실행하여 더 많은 이미지를 캡처하는 것을 권장합니다.\n")

    if gray is None:
        print("이미지가 하나도 캡처되지 않아 보정을 수행할 수 없습니다.")
        return

    # 이미지 크기 (width, height)
    image_size = gray.shape[::-1]

    # 보정 실행
    reprojection_error, camera_matrix, dist_coeffs = run_calibration(
        objpoints, imgpoints, image_size
    )

    # JSON 저장 (fx, fy, cx, cy 형태)
    save_to_json(
        camera_matrix,
        dist_coeffs,
        image_size,
        PATTERN_SIZE,
        SQUARE_SIZE,
        reprojection_error,
    )

    print("\n=== 카메라 보정 완료 ===")


if __name__ == "__main__":
    main()
