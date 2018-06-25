import math
import numpy as np

TWO_PI = 2 * math.pi


def get_not_convex_point(polygon):
    # Check for too few points
    if len(polygon) < 3:
        raise ValueError
    # Get starting information
    old_x, old_y = polygon[-2]
    new_x, new_y = polygon[-1]
    new_direction = math.atan2(new_y - old_y, new_x - old_x)
    angle_sum = 0.0
    pos_angle_list = []
    neg_angle_list = []
    # Check each point (the side ending there, its angle) and accum. angles
    for ndx, new_point in enumerate(polygon):
        # Update point coordinates and side directions, check side length
        old_x, old_y, old_direction = new_x, new_y, new_direction
        new_x, new_y = new_point
        new_direction = math.atan2(new_y - old_y, new_x - old_x)
        if old_x == new_x and old_y == new_y:
            raise ValueError  # repeated consecutive points
        # Calculate & check the normalized direction-change angle
        angle = new_direction - old_direction
        if angle <= -math.pi:
            angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
        elif angle > math.pi:
            angle -= TWO_PI
        if angle < 0.0:
            neg_angle_list.append(ndx - 1)
        else:
            pos_angle_list.append(ndx - 1)
        # Accumulate the direction-change angle
        angle_sum += angle
    if len(pos_angle_list) < len(neg_angle_list):
        return pos_angle_list, neg_angle_list
    else:
        return neg_angle_list, pos_angle_list


def split_signs(contour, non_convex_points):
    if 2 < len(non_convex_points) <= 4:
        min_dist = 8000
        closest_points = None
        for i, p1 in enumerate(non_convex_points):
            for p2 in non_convex_points[i + 1:]:
                dist = np.linalg.norm(contour[p1] - contour[p2], 2)
                if min_dist > dist and len(contour) - 3 > abs(p1 - p2) > 3:
                    min_dist = dist
                    closest_points = [p1, p2]
                    print(p1, p2)
        # if the convex points are to close to each other, it's probably false positive
        if closest_points is None:
            return []
        non_convex_points = closest_points
    elif len(non_convex_points) != 2:
        print("Contour with not exactly 2 non convex point ({} points) are not handled yet...".format(
            len(non_convex_points)))
        return []
    convex_contour_list = []
    poly_1 = np.concatenate([contour[:non_convex_points[0] + 1], contour[non_convex_points[-1]:]], axis=0)
    convex_contour_list.append(poly_1)
    poly_2 = contour[non_convex_points[0]:non_convex_points[-1] + 1]
    convex_contour_list.append(poly_2)
    return convex_contour_list


def try_to_split_signs(contour):
    shaped_contour = contour.reshape((-1, 2))
    non_convex_points, convex_point = get_not_convex_point(shaped_contour)
    if len(non_convex_points) == 1:
        return [shaped_contour[convex_point]]
    else:
        return split_signs(shaped_contour, non_convex_points)



