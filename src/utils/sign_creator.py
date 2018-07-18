import cv2


def speed_sign_creator(speed: int, path_to_base_sign="sign/speed_limit/W13-1P_out-01.png"):
    ss = cv2.imread(path_to_base_sign)
    ss = cv2.putText(ss, "{:02d}".format(speed), (40, 135), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 0, 15)
    return ss
