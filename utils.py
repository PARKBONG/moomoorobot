# Suction 그리퍼 함수 정의
def pump_on(cobot):
    """공압 펌프 ON (물체 흡착)"""
    cobot.set_basic_output(5, 0) # pump on
    cobot.set_basic_output(2, 0) # vent?

def pump_off(cobot):
    """공압 펌프 OFF (물체 방출)"""
    cobot.set_basic_output(5, 1)
    cobot.set_basic_output(2, 1)