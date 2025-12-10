import time

# from api.april_tag_tracker import AprilTagTracker # 경로따라 달라짐!
from april_tag_tracker import AprilTagTracker

# handle 선언
# 핸들이 뭐냐? 그냥 내가 이름 지음ㅋ
# 그냥 문법이니까 외우셈 

# 세줄 요약
# 이렇게 쓰면 된다
# ====================================
# tracker = AprilTagTracker()
# tracker.add_tag(ID=5, size_cm=2)
# object_pose = tracker.get_pose(ID=5) 
# ====================================


tracker = AprilTagTracker(
    pop_window=False, # 이거 바꿔보셈
    intrinsic_path="./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json",
    extrinsic_path="./../localization_april_tag/extrinsic_calibration_result.json",
)

OBJECT_TAG = 5
OBJECT_TAG_SIZE_CM = 2.0
TARGET_TAG = 6
TARGET_TAG_SIZE_CM = 6.0

# 태그 등록
tracker.add_tag(ID=OBJECT_TAG, size_cm=OBJECT_TAG_SIZE_CM)  # 작은 태그
tracker.add_tag(ID=TARGET_TAG, size_cm=TARGET_TAG_SIZE_CM)  # 큰 태그

# 정적 업데이트: 물체가 안움직이는 경우. =========================================================
# 혹은 이걸 필요할때마다, 혹은 자주 불러줘도 됨.
# 연산량이 작고 사용법 간단하기 때문에 추후 april tag랑 다른 코드(예:로봇, 등) 이어붙일때는 이거를 추천함. 
# 이런식으로 쓸때는 pop_window=False줘야댐. 필요하면 추가해줄테니까말하셈.
object_pose = tracker.get_pose(OBJECT_TAG) 
print("Object Tag ID:5 pose:", object_pose)

target_pose = tracker.get_pose(TARGET_TAG)
print("Target Tag ID:6 pose:", target_pose)
# ===========================================================================================


# # 동적 업데이트: 물체가 움직이는 경우.============================================================
# 이런식으로 쓸때는 pop_window=True줘도 됨. 한번돌려보셈.
while True:
    object_pose = tracker.get_pose(OBJECT_TAG)

    if object_pose:
        x, y, z, rx, ry, rz = object_pose
        print("[Object 5] x=%.3f y=%.3f z=%.3f  rx=%.1f ry=%.1f rz=%.1f"
              % (x, y, z, rx, ry, rz))

    target_pose = tracker.get_pose(TARGET_TAG)
    if target_pose:
        x, y, z, rx, ry, rz = target_pose
        print("[Target 6] x=%.3f y=%.3f z=%.3f  rx=%.1f ry=%.1f rz=%.1f"
              % (x, y, z, rx, ry, rz))

    
    # 무한루프 탈출용.
    # pop_window=False로 하면 필요없음 (햇갈리니까 그냥둬도댐)
    # pop_window=True인데 이거없으면 무슨일 일어나는지 한번 해보셈.
    if tracker.is_pressed_esc():
        print("ESC 눌러서 종료함.")
        break
    
    time.sleep(0.02)

# ===========================================================================================

# 태그 삭제 (더이상 특정 태그 필요없는 경우. 그냥 프로그램 끄는거면 선언안해도 됨.)
tracker.remove_tag(OBJECT_TAG)
tracker.remove_tag(TARGET_TAG)

tracker.get_pose(OBJECT_TAG)  # 테스트용. None 반환