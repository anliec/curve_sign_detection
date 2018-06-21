import cv2
import numpy as np

from src.detector.concave_sign_spliter import try_to_split_signs


def find_sign_contour(mask: np.ndarray):
    contours = detect_mask_contour(mask)
    filtered_contours = list(filter(can_contour_be_sign, contours))
    approximated_contours = approximate_contours(filtered_contours)
    return approximated_contours


def detect_mask_contour(mask: np.ndarray):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def can_contour_be_sign(c):
    perimeter = cv2.arcLength(c, True)
    # print("Perimeter:", perimeter)
    if perimeter < 110:
        return False
    area = cv2.contourArea(c)
    # print("Area:", area)
    squareness = perimeter**2 / (16 * area)  # for a square, perimeter**2 / (16 * area) should be equal to 1.0
    # print("Ratio:", squareness)
    if squareness > 3.0:
        return False
    height = np.min(c.reshape((-1, 2)), axis=0)[1]
    h_ratio = height**2 / area  # small sign are usually on top of the image and big one in the middle
    # print("height ratio:", h_ratio)
    if h_ratio > 250:  # filter small sign too low in the image
        return False
    return True


def approximate_contours(contour_list, block_recursion: bool=False):
    approx_contours = []
    for c in contour_list:
        if len(c) == 0:
            continue
        approx_c = cv2.approxPolyDP(c, 0.012 * cv2.arcLength(c, True), True)
        # print(len(approx_c))
        if len(approx_c) == 4:
            cc = cv2.convexHull(c)
            approx_c = cv2.approxPolyDP(cc, 0.065 * cv2.arcLength(cc, True), True)
            approx_contours.append(approx_c)
        elif 4 < len(approx_c) < 7:
            rect = cv2.minAreaRect(approx_c)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area * 0.75 < cv2.contourArea(approx_c):
                m = cv2.moments(c)
                center = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
                # print(center)
                rect = (center, (rect[1][0] * 0.9, rect[1][1] * 0.9), rect[2])
                box = cv2.boxPoints(rect)
                box = np.int0(box).reshape((4, 1, 2))
                approx_contours.append(box)
        elif 12 > len(approx_c) > 6 and not block_recursion:
            # probably two sign merged in one...
            splitted_contours = try_to_split_signs(approx_c)
            approx_contours += approximate_contours(splitted_contours, block_recursion=True)
    return approx_contours





