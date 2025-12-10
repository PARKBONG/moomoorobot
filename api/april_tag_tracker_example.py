import time

# from api.april_tag_tracker import AprilTagTracker # 경로따라 달라짐!
from april_tag_tracker import AprilTagTracker

# handle 선언
# 핸들이 뭐냐? 그냥 내가 이름 지음ㅋ
# 그냥 문법이니까 외우셈 

object_handle = AprilTagTracker(
    ID=5,
    size_cm=2.0,
    pop_window=False,
    intrinsic_path="./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json",
    extrinsic_path="./../localization_april_tag/extrinsic_calibration_result.json",
)

target_handle = AprilTagTracker(
    ID=6,
    size_cm=6.0,
    pop_window=False,
    intrinsic_path="./../camera_intrinsic_estimation/intrinsic_calibration_result_20251210_115333.json",
    extrinsic_path="./../localization_april_tag/extrinsic_calibration_result.json",
)

# 정적 업데이트: 물체가 안움직이는 경우.
# 혹은 이걸 필요할때마다, 혹은 자주 불러줘도 됨.
# 연산량이 작고 사용법 간단하기 때문에 추후 april tag랑 다른 코드(예:로봇, 등) 이어붙일때는 이거를 추천함. 
object_pose = object_handle.get_pose() # get_pose()함수: 너가 유일하게 알면 되는 함수.
print("Object Pose:", object_pose) 

# 동적 업데이트: 물체가 움직이는 경우
while True:
    object_pose = object_handle.get_pose()
    if object_pose is not None:
        x, y, z, rx, ry, rz = object_pose
        print(
            "[Object Pose] x: %.3f m, y: %.3f m, z: %.3f m, rx: %.3f deg, ry: %.3f deg, rz: %.3f deg"
            % (x, y, z, rx, ry, rz)
        )
        time.sleep(0.1)
        
    target_pose = target_handle.get_pose()
    if target_pose is not None:
        x, y, z, rx, ry, rz = target_pose
        print(
            "[Target Pose] x: %.3f m, y: %.3f m, z: %.3f m, rx: %.3f deg, ry: %.3f deg, rz: %.3f deg"
            % (x, y, z, rx, ry, rz)
        )
        time.sleep(0.01)
        
        