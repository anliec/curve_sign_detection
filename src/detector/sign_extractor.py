import cv2
import numpy as np

BOX_SIZE = 100
pts_dst_anti_horaire = np.float32([[0, BOX_SIZE], [BOX_SIZE, BOX_SIZE], [BOX_SIZE, 0], [0, 0]])
pts_dst_horaire = np.float32([[0, BOX_SIZE], [0, 0], [BOX_SIZE, 0], [BOX_SIZE, BOX_SIZE]])


def extract_sign_image(contours, image: np.ndarray):
    sign_images = []
    for c in contours:
        if len(c) == 4:
            pts_src = np.array(c, dtype=np.float32).reshape((4, 2))
            # ensure rotation is the same for every sign
            right_angle, bottom_angle = np.argmax(pts_src, axis=0)
            diff_to_bottom = pts_src - pts_src[bottom_angle]
            bottom_threshold = (pts_src[bottom_angle][1] - np.min(pts_src[:, 1])) * 0.2
            mask_bt_corner = diff_to_bottom[:, 1] < -bottom_threshold
            if mask_bt_corner.sum() < 3:  # if two corner are near the bottom we only take the left most one
                masked_corner = pts_src.copy()
                masked_corner[mask_bt_corner, :] = 8000  # mask out the others corner
                sorted_arg = np.argsort(masked_corner[:, 0])  # get the position of each corner sorted by x value
                bottom_angle = sorted_arg[0]
                right_angle = sorted_arg[1]
            pts_src = np.concatenate([pts_src[bottom_angle:], pts_src[:bottom_angle]])
            if right_angle == bottom_angle + 1:
                pts_dst = pts_dst_anti_horaire
            else:
                pts_dst = pts_dst_horaire
            # extract the sign
            transform_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            sign = cv2.warpPerspective(image, transform_matrix, (BOX_SIZE, BOX_SIZE))
            sign_images.append(sign)
    return sign_images

