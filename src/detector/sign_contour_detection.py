import cv2
import numpy as np

from src.detector.concave_sign_spliter import try_to_split_signs

contours_threshold = {'min_per': 70,
                      'width_ratio': 3.0,
                      'bounding_rect_area_max_ratio': 0.5,
                      'max_squareness': 3.0,
                      'max_height_ratio': 170}

strict_contours_threshold = {'min_per': 50,
                             'width_ratio': 5.0,
                             'bounding_rect_area_max_ratio': 0.7,
                             'max_squareness': 1.5,
                             'max_height_ratio': 200}


def find_sign_contour(mask: np.ndarray):
    contours = detect_mask_contour(mask)
    contours = detect_and_merge_triangles(contours)
    filtered_contours = list(filter(can_contour_be_sign, contours))
    # print(len(filtered_contours))
    approximated_contours = approximate_contours(filtered_contours)
    approximated_contours = list(filter(lambda x: can_contour_be_sign(x, strict=True), approximated_contours))
    # print(len(approximated_contours))
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
    bounding_box = cv2.minAreaRect(c)
    if bounding_box[1][0] > param['width_ratio'] * bounding_box[1][1]:
        return False
    area = cv2.contourArea(c)
    if area < param['bounding_rect_area_max_ratio'] * bounding_box[1][0] * bounding_box[1][1]:
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
        if 4 <= len(approx_c) < 8:
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
            if len(approx_cc) == 4 and ((approx_accuracy > 0.8 and not block_recursion) or approx_accuracy > 0.9):
                approx_contours.append(approx_cc)
                continue
            elif rect_accuracy > 0.7:
                m = cv2.moments(c)
                center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
                rect = (center, (rect[1][0] * 1, rect[1][1] * 1), rect[2])
                box = cv2.boxPoints(rect)
                box = np.int0(box).reshape((4, 1, 2))
                approx_contours.append(box)
                continue
        if 6 < len(approx_c) < 14 and not block_recursion:
            # probably two sign merged in one...
            splitted_contours = try_to_split_signs(approx_c)
            approx_contours += approximate_contours(splitted_contours, block_recursion=True)
    return approx_contours


def detect_and_merge_triangles(contours):
    triangle_list = []
    triangle_full_contour_list = []
    non_triangle_list = []
    for c in contours:
        approx_c = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
        # print(len(approx_c))
        if len(approx_c) == 3 or (len(approx_c) == 4 and cv2.arcLength(approx_c, True) < 100):
            triangle_list.append(approx_c)
            triangle_full_contour_list.append(c)
        else:
            non_triangle_list.append(c)

    triangle_dsc = []
    for i, t in enumerate(triangle_list):
        triangle_dsc.append({'center': np.mean(t, axis=0),
                             'size': cv2.arcLength(t, True),
                             'triangle_id': i})

    arrow_sign_contours = []
    matched_ids = []
    for desc in triangle_dsc:
        if desc['triangle_id'] in matched_ids:
            continue
        center = desc['center']
        search_radius = int(desc['size'] * 0.7)
        matches = []
        for other_desc in triangle_dsc:
            if np.linalg.norm(center - other_desc['center'], 1) < search_radius:
                # if size are similar (prevent matching sign with noise)
                if other_desc['size'] < 3 * desc['size'] and other_desc['size'] * 3 > desc['size']:
                    matches.append(other_desc['triangle_id'])
        # print(search_radius, len(matches))
        if len(matches) == 3:
            all_pts = np.concatenate([triangle_list[m] for m in matches], axis=0)
            sign_contour = cv2.convexHull(all_pts)
            arrow_sign_contours.append(sign_contour)
            matched_ids += matches

    # print(len(arrow_sign_contours))

    for i, c in enumerate(triangle_full_contour_list):
        if i not in matched_ids:
            non_triangle_list.append(c)

    return arrow_sign_contours + non_triangle_list


