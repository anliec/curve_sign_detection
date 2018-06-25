import cv2
import numpy as np

from src.detector.concave_sign_spliter import try_to_split_signs

contours_threshold = {'min_per': 70,
                      'width_ratio': 3.0,
                      'bounding_rect_area_max_ratio': 0.4,
                      'max_squareness': 3.0,
                      'max_height_ratio': 160}

strict_contours_threshold = {'min_per': 50,
                             'width_ratio': 5.0,
                             'bounding_rect_area_max_ratio': 0.4,
                             'max_squareness': 1.5,
                             'max_height_ratio': 500}


def find_sign_contour(mask: np.ndarray):
    contours = detect_mask_contour(mask)
    filtered_contours = list(filter(can_contour_be_sign, contours))
    print(len(filtered_contours))
    approximated_contours = approximate_contours(filtered_contours)
    approximated_contours = list(filter(lambda x: can_contour_be_sign(x, strict=True), approximated_contours))
    print(len(approximated_contours))
    return approximated_contours


def detect_mask_contour(mask: np.ndarray):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def can_contour_be_sign(c, strict: bool=False):
    if not strict:
        param = contours_threshold
    else:
        param = strict_contours_threshold
    perimeter = cv2.arcLength(c, True)
    if perimeter < param['min_per']:
        return False
    bounding_box = cv2.boundingRect(c)
    if bounding_box[2] > param['width_ratio'] * bounding_box[3]:
        return False
    area = cv2.contourArea(c)
    if area < param['bounding_rect_area_max_ratio'] * bounding_box[2] * bounding_box[3]:
        return False
    squareness = perimeter**2 / (16 * area)  # for a square, perimeter**2 / (16 * area) should be equal to 1.0
    if squareness > param['max_squareness']:
        return False
    height = np.min(c.reshape((-1, 2)), axis=0)[1]
    h_ratio = height**2 / area  # small sign are usually on top of the image and big one in the middle
    if h_ratio > param['max_height_ratio']:  # filter small sign too low in the image
        return False
    return True


def approximate_contours(contour_list, block_recursion: bool=False):
    approx_contours = []
    for c in contour_list:
        if len(c) == 0:
            continue
        approx_c = cv2.approxPolyDP(c, 0.017 * cv2.arcLength(c, True), True)
        # print('approx:', len(approx_c))
        if 4 <= len(approx_c) < 9:
            # convert to a convex polynome
            cc = cv2.convexHull(c)
            # approximate by reducing the number of line in polygone
            approx_cc = cv2.approxPolyDP(cc, 0.065 * cv2.arcLength(cc, True), True)
            # approximate by taking the smallest possible rect enclosing the polygone
            rect = cv2.minAreaRect(approx_c)
            # choose the one that have the best accuracy
            rect_area = rect[1][0] * rect[1][1]
            approx_cc_area = cv2.contourArea(approx_cc)
            cc_area = cv2.contourArea(cc)
            rect_accuracy = cc_area / rect_area
            approx_accuracy = approx_cc_area / cc_area
            if len(approx_cc) == 4 and ((approx_accuracy > 0.8 and not block_recursion) or approx_accuracy > 0.95):
                approx_contours.append(approx_cc)
                continue
            elif rect_accuracy > 0.75:
                m = cv2.moments(c)
                center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
                rect = (center, (rect[1][0] * 1, rect[1][1] * 1), rect[2])
                box = cv2.boxPoints(rect)
                box = np.int0(box).reshape((4, 1, 2))
                approx_contours.append(box)
                continue
        if 14 > len(approx_c) > 6 and not block_recursion:
            # probably two sign merged in one...
            splitted_contours = try_to_split_signs(approx_c)
            approx_contours += approximate_contours(splitted_contours, block_recursion=True)
    return approx_contours





