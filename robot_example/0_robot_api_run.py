# MyCobot 280 필수 API 예제 모음
# Essential API Examples for MyCobot 280
import time
from pymycobot import MyCobot280

from utils import pump_on, pump_off

# GPT가 짠 거임
# 안이쁜 모션이 있긴 한데 참고로는 문제없을듯
# 로봇 안넘어지게 해놓고 걍 돌려보셈

if __name__ == "__main__":
    # ==================== 1. 연결 및 초기화 ====================
    print("=== 1. 로봇 연결 및 초기화 ===")
    cobot = MyCobot280('COM6')  # 포트 번호 확인 필요
    time.sleep(1)
    
    # 시스템 버전 확인
    version = cobot.get_system_version()
    print(f"시스템 버전: {version}")
    
    # 전원 상태 확인 및 켜기
    power_status = cobot.is_power_on()
    print(f"전원 상태: {power_status} (1=ON, 0=OFF)")
    if power_status != 1:
        cobot.power_on()
        print("전원 켜기 완료")
        time.sleep(1)
    
    # 컨트롤러 연결 확인
    controller_connected = cobot.is_controller_connected()
    print(f"컨트롤러 연결: {controller_connected}")
    
    # ==================== 2. 홈 포지션 설정 ====================
    print("\n=== 2. 홈 포지션으로 이동 ===")
    
    # Fresh mode 설정 (0: 기존 명령 완수 후 새 명령, 1: 새 명령 즉시 실행)
    cobot.set_fresh_mode(0)
    time.sleep(0.5)
    
    # 홈 포지션 각도 정의 (표준 홈 포즈)
    HOME_POSE_ANGLES = [0, 0, -90, 0, 0, 0]
    
    # 홈 포지션으로 이동
    print(f"홈 포지션으로 이동: {HOME_POSE_ANGLES}")
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    time.sleep(3)
    
    # 홈 포지션의 Cartesian 좌표 확인
    HOME_POSE_XYZ = cobot.get_coords()
    print(f"홈 포지션 Cartesian 좌표: {HOME_POSE_XYZ}")
    print(f"  X={HOME_POSE_XYZ[0]:.1f}, Y={HOME_POSE_XYZ[1]:.1f}, Z={HOME_POSE_XYZ[2]:.1f}")
    print(f"  RX={HOME_POSE_XYZ[3]:.2f}, RY={HOME_POSE_XYZ[4]:.2f}, RZ={HOME_POSE_XYZ[5]:.2f}")
    
    # ==================== 3. 각도(Joint) 제어 ====================
    print("\n=== 3. 각도 제어 (Joint Control) ===")
    
    # 현재 각도 읽기
    current_angles = cobot.get_angles()
    print(f"현재 각도: {current_angles}")
    
    # 각 관절의 최소/최대 각도 확인
    print("\n관절 각도 범위:")
    for i in range(1, 7):
        min_angle = cobot.get_joint_min_angle(i)
        max_angle = cobot.get_joint_max_angle(i)
        print(f"  Joint {i}: {min_angle}° ~ {max_angle}°")
    time.sleep(1)
    
    # 개별 관절 제어 예제
    print("\nJoint 1을 45도로 이동")
    cobot.send_angle(1, 45, 30)
    time.sleep(3)
    
    print("Joint 1을 다시 0도로")
    cobot.send_angle(1, 0, 30)
    time.sleep(3)
    
    # 홈 포지션으로 복귀
    print("홈 포지션으로 복귀")
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    time.sleep(3)
    
    # ==================== 4. Cartesian 좌표 제어 ====================
    print("\n=== 4. Cartesian 좌표 제어 ===")
    
    # 현재 좌표 읽기
    current_coords = cobot.get_coords()
    print(f"현재 좌표: {current_coords}")
    print(f"  X={current_coords[0]:.1f}, Y={current_coords[1]:.1f}, Z={current_coords[2]:.1f}")
    print(f"  RX={current_coords[3]:.2f}, RY={current_coords[4]:.2f}, RZ={current_coords[5]:.2f}")
    
    # 좌표로 이동 (오리엔테이션 유지하며 위치만 변경)
    print("\n좌표로 이동 (오리엔테이션 유지)")
    # 홈 포지션의 오리엔테이션 유지
    cobot.send_coords([200, 0, 200, HOME_POSE_XYZ[3], HOME_POSE_XYZ[4], HOME_POSE_XYZ[5]], 30, mode=1)
    time.sleep(3)
    
    # 개별 좌표 축 제어 (Z축만 변경, 오리엔테이션 유지)
    print("Z축만 150mm로 변경 (오리엔테이션 유지)")
    cobot.send_coord(3, 150, 30)
    time.sleep(3)
    
    # 홈 포지션으로 복귀
    print("홈 포지션으로 복귀")
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    time.sleep(3)
    
    # ==================== 5. 서보 모터 상태 확인 ====================
    print("\n=== 5. 서보 모터 상태 ===")
    
    # 모든 서보 활성화 상태 확인
    all_servo_enabled = cobot.is_all_servo_enable()
    print(f"모든 서보 활성화 상태: {all_servo_enabled}")
    
    # 개별 서보 상태 확인
    for i in range(1, 7):
        servo_status = cobot.is_servo_enable(i)
        print(f"  Servo {i}: {servo_status} (1=활성화, 0=비활성화)")
    
    # ==================== 6. 움직임 상태 확인 ====================
    print("\n=== 6. 움직임 상태 확인 ===")
    
    # 로봇이 움직이는지 확인
    test_angles = [0, 10, -80, 10, 0, 0]
    cobot.send_angles(test_angles, 20)
    time.sleep(0.5)
    is_moving = cobot.is_moving()
    print(f"로봇 움직임 중: {is_moving} (1=움직임, 0=정지)")
    time.sleep(2)
    
    # 목표 위치 도달 확인
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    time.sleep(3)
    in_position = cobot.is_in_position(HOME_POSE_ANGLES, 0)  # 0=각도 기준
    print(f"홈 포지션 도달: {in_position}")
    
    # ==================== 7. JOG 제어 ====================
    print("\n=== 7. JOG 제어 (미세 조정) ===")
    
    # Joint 1을 증가 방향으로 조그
    print("Joint 1 증가 방향 JOG")
    cobot.jog_angle(1, 1, 30)  # 1=증가, 0=감소
    time.sleep(1)
    cobot.stop()
    
    # Joint 1을 감소 방향으로 조그
    print("Joint 1 감소 방향 JOG")
    cobot.jog_angle(1, 0, 30)
    time.sleep(1)
    cobot.stop()
    
    # 증분 이동 (정확한 각도만큼 이동)
    print("Joint 1을 15도 증분 이동")
    cobot.jog_increment(1, 15, 30)
    time.sleep(3)
    
    # 홈 포지션으로 복귀
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    time.sleep(3)
    
    # ==================== 8. 일시정지/재개/정지 ====================
    print("\n=== 8. 일시정지/재개/정지 ===")
    
    # 움직임 시작
    cobot.send_angles([20, 20, -70, 20, 0, 0], 20)
    time.sleep(1)
    
    # 일시정지
    print("일시정지")
    cobot.pause()
    time.sleep(1)
    
    # 일시정지 상태 확인
    paused = cobot.is_paused()
    print(f"일시정지 상태: {paused}")
    
    # 재개
    print("재개")
    cobot.resume()
    time.sleep(2)
    
    # 정지
    print("정지")
    cobot.stop()
    time.sleep(1)
    
    # 홈 포지션으로 복귀
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    time.sleep(3)
    
    # ==================== 9. 서보 릴리즈 (탈력) ====================
    # 봉경: 이거 쓰지 말것. 쓰면 모터 힘 전부 풀리면서 로봇 주저앉음. 
    # 존재 목적은 데이터 수집임 -- 사람이 손으로 움직이고, 로봇이 따라가게 하는 목적임.
    # 써보고 싶으면 로봇 안무너지게 손으로 잡을것
    # print("\n=== 9. 서보 릴리즈 ===")
    # cobot.release_all_servos()
    # time.sleep(2)
    # cobot.focus_all_servos()
    # time.sleep(1)
    
    # ==================== 10. LED 제어 ====================
    print("\n=== 10. LED 색상 제어 ===")
    
    # 빨간색
    print("LED: 빨간색")
    cobot.set_color(255, 0, 0)
    time.sleep(1)
    
    # 초록색
    print("LED: 초록색")
    cobot.set_color(0, 255, 0)
    time.sleep(1)
    
    # 파란색
    print("LED: 파란색")
    cobot.set_color(0, 0, 255)
    time.sleep(1)
    
    # LED 끄기
    print("LED: 끄기")
    cobot.set_color(0, 0, 0)
    
    # ==================== 11. 공압 Suction 그리퍼 제어 ====================
    print("\n=== 11. 공압 Suction 그리퍼 제어 ===")
    
    # 펌프 OFF (초기 상태)
    print("펌프 OFF")
    pump_off(cobot)
    time.sleep(3)
    
    # 펌프 ON (물체 흡착)
    print("펌프 ON - 물체 흡착")
    pump_on(cobot)
    time.sleep(3)
    
    # 펌프 OFF (물체 방출)
    print("펌프 OFF - 물체 방출")
    pump_off(cobot)
    time.sleep(3)
    
    # ==================== 12. 엔코더 값 읽기 ====================
    print("\n=== 12. 엔코더 제어 ===")
    
    # 모든 엔코더 값 읽기
    encoders = cobot.get_encoders()
    print(f"현재 엔코더 값: {encoders}")
    
    # 개별 엔코더 값 읽기
    encoder_1 = cobot.get_encoder(1)
    print(f"Joint 1 엔코더: {encoder_1}")
    
    # ==================== 13. 홈 포지션으로 복귀 ====================
    print("\n=== 13. 홈 포지션으로 복귀 ===")
    cobot.send_angles(HOME_POSE_ANGLES, 30)
    time.sleep(3)
    
    # 최종 위치 확인
    final_angles = cobot.get_angles()
    final_coords = cobot.get_coords()
    print(f"최종 각도: {final_angles}")
    print(f"최종 좌표: {final_coords}")
    
    print("\n=== 모든 API 테스트 완료 ===")
    print("\n주요 API 요약:")
    print("  - 각도 제어: get_angles(), send_angle(), send_angles()")
    print("  - 좌표 제어: get_coords(), send_coord(), send_coords()")
    print("  - 상태 확인: is_moving(), is_in_position(), is_paused()")
    print("  - 서보 제어: release_servo(), focus_servo(), is_servo_enable()")
    print("  - JOG 제어: jog_angle(), jog_coord(), jog_increment()")
    print("  - 제어: pause(), resume(), stop()")
    print("  - LED: set_color()")
    print("  - 공압 펌프: set_basic_output(5, 0/1)")
    print("  - 전원: power_on(), power_off(), is_power_on()")
