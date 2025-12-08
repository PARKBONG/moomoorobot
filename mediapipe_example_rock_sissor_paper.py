# pip install pyrealsense2 mediapipe opencv-python numpy 

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# 내가 돌려봤는데 보자기가 잘 안됨
# 너가 직접 개발할필요는 없고, 해당 사이트 참고. 
# https://github.com/kairess/Rock-Paper-Scissors-Machine
# 뻐큐 필터가 만들기 쉬워보이는데, 이정도는 너가 직접 만들어보셈

# ---------------------------
#  Hand gesture classifier
# ---------------------------
def classify_rps(hand_landmarks, handedness_label):
    """
    MediaPipe Hand landmarks로부터
    Rock / Paper / Scissors / Unknown 을 판정.
    handedness_label: "Left" or "Right"
    """
    # MediaPipe index reference:
    # 0: wrist
    # Thumb: 1,2,3,4 (tip=4)
    # Index: 5,6,7,8 (tip=8)
    # Middle: 9,10,11,12 (tip=12)
    # Ring: 13,14,15,16 (tip=16)
    # Pinky: 17,18,19,20 (tip=20)

    lm = hand_landmarks

    # 편의를 위해 numpy 배열로 전환
    coords = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)

    # 이미지 좌표계: (0,0)이 좌상단, y 커질수록 아래
    # -> 손가락이 "펴진" 경우: tip.y < pip.y (즉 더 위에 있음)
    def is_finger_extended(tip_idx, pip_idx, y_thresh=0.02):
        tip_y = coords[tip_idx, 1]
        pip_y = coords[pip_idx, 1]
        return (pip_y - tip_y) > y_thresh

    # 엄지 판단 (좌/우 손 다름)
    # Right hand: 엄지 펴지면 tip.x > ip.x
    # Left hand:  엄지 펴지면 tip.x < ip.x
    def is_thumb_extended(thresh=0.02):
        tip_x = coords[4, 0]
        ip_x = coords[3, 0]

        if handedness_label == "Right":
            return (tip_x - ip_x) > thresh
        else:  # "Left"
            return (ip_x - tip_x) > thresh

    thumb = is_thumb_extended()
    index_f = is_finger_extended(8, 6)
    middle_f = is_finger_extended(12, 10)
    ring_f = is_finger_extended(16, 14)
    pinky_f = is_finger_extended(20, 18)

    fingers = [thumb, index_f, middle_f, ring_f, pinky_f]
    num_extended = sum(fingers)

    # ----- Rule-based classification -----
    # Rock: 거의 모든 손가락이 접혀있음
    if num_extended == 0:
        return "Rock"

    # Paper: 다 펴져있음
    if all(fingers):
        return "Paper"

    # Scissors: index + middle만 펴지고 나머지는 접힘 (엄지는 상관 덜 봄)
    if index_f and middle_f and not ring_f and not pinky_f and (not thumb or thumb):
        # 엄지는 약간 애매해도 허용
        return "Scissors"

    return "Unknown"


def main():
    # ---------------------------
    # RealSense D405 설정
    # ---------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # Color만 사용 (원하면 depth도 enable_stream 추가 가능)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # ---------------------------
    # MediaPipe Hands 설정
    # ---------------------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # MediaPipe는 RGB 입력 권장
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            gesture_text = "No Hand"

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label  # "Left" or "Right"

                # 제스처 분류
                gesture_text = classify_rps(hand_landmarks.landmark, handedness)

                # 랜드마크 표시
                mp_drawing.draw_landmarks(
                    color_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            # 화면에 결과 텍스트 출력
            cv2.putText(
                color_image,
                f"Gesture: {gesture_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("D405 + MediaPipe RPS", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        pipeline.stop()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
