# MyCobot 280 필수 API 예제 모음
# Essential API Examples for MyCobot 280
import time
from pymycobot import MyCobot280
from utils import pump_on, pump_off

if __name__ == "__main__":
    
    cobot = MyCobot280('COM6') 
    cobot.set_fresh_mode(1) # 1로 해야대나? 잘모르겠음
    
    # 일단 홈포즈로 움직이는게 표준임
    HOME_POSE_ANGLES = [0, 0, -90, 0, 0, 0]
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    HOME_POSE_XYZ = cobot.get_coords()
    print(f"현재 Cartesian: {HOME_POSE_XYZ}")
    time.sleep(2)

    dy = 20
    target_pose_xyz = HOME_POSE_XYZ.copy()
    target_pose_xyz[1] += dy
    cobot.send_coords(target_pose_xyz, 30)
    print(f"current Cartesian after Y+10mm: {cobot.get_coords()}")