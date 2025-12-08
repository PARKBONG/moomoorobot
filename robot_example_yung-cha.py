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

    print(f"현재 조인트 각도: {HOME_POSE_ANGLES}")
    print(f"현재 좌표: {HOME_POSE_XYZ}")
    print(f"  X={HOME_POSE_XYZ[0]}, Y={HOME_POSE_XYZ[1]}, Z={HOME_POSE_XYZ[2]}")
    print(f"  RX={HOME_POSE_XYZ[3]}, RY={HOME_POSE_XYZ[4]}, RZ={HOME_POSE_XYZ[5]}")

    dy = 30
    target_pose_xyz = HOME_POSE_XYZ.copy()
    target_pose_xyz[1] += dy
    cobot.send_coords(target_pose_xyz, 30)
    print("Y축만 30mm 변경")

    dz = -100  # Z축으로 200mm 내리기
    target_pose_xyz[2] += dz
    cobot.send_coords(target_pose_xyz, 30) 
    print("Z축만 200mm로 변경")

    time.sleep(2)
    pump_on(cobot)
    time.sleep(3)
    print("물체 흡착 완료")

    target_pose = [130, 80, HOME_POSE_XYZ[2], HOME_POSE_XYZ[3], HOME_POSE_XYZ[4], HOME_POSE_XYZ[5]]
    cobot.send_coords(target_pose, 30) 
    print("이영~ 차!")

    pump_off(cobot)
    time.sleep(3)
    print("물체 흡착 해제 완료")

    # See what happens
    # cobot.send_angles(HOME_POSE_ANGLES, 30)
    # ry = 30
    # target_pose_xyz = HOME_POSE_XYZ.copy()
    # target_pose_xyz[4] += ry
    # cobot.send_coords(target_pose_xyz, 30)
    # print("RY축만 30도 변경")
    # 로봇 XYZ "이동"은 안어려운데 "회전"은 너가 따로 시간내서 공부해야댐 궁금하면 물어보셈. 공부하기 싫으면 일단 RY만 쓰기를...
    # 이거 그냥 반복하면 너가 원하는 대로 다 움직일 수 있음.
    # 로봇 움직이다가 이상한 포즈로 가서 더이상 명령 안먹는다 -> 역기구학 못풀거나 조인트 한계 벗어났을 가능성이 제일큼. 그다음 통신(소프트웨어 혹은 케이블). GPT한테 물어봐

    # 일단 홈포즈로 움직이는게 표준임
    HOME_POSE_ANGLES = [0, 0, -90, 0, 0, 0]
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    HOME_POSE_XYZ = cobot.get_coords()