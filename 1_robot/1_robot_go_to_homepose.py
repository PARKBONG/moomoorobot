from pymycobot import MyCobot280

cobot = MyCobot280('COM6') 
cobot.set_fresh_mode(0)

# 일단 홈포즈로 움직이는게 표준임
HOME_POSE_ANGLES = [0, 0, -90, 0, 0, 0]
cobot.send_angles(HOME_POSE_ANGLES, 30)
HOME_POSE_XYZ = cobot.get_coords()

print(f"현재 조인트 각도: {HOME_POSE_ANGLES}")
print(f"현재 좌표: {HOME_POSE_XYZ}")
print(f"  X={HOME_POSE_XYZ[0]}, Y={HOME_POSE_XYZ[1]}, Z={HOME_POSE_XYZ[2]}")
print(f"  RX={HOME_POSE_XYZ[3]}, RY={HOME_POSE_XYZ[4]}, RZ={HOME_POSE_XYZ[5]}")
