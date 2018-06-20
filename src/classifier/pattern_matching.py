import cv2
import numpy as np
import glob
import os

from src.detector.sign_extractor import BOX_SIZE


def match_sign_with_pattern(sign_image_list, ref_sign_path: str="sign"):
    result_list = []
    for hsv_sign in sign_image_list:
        # hsv_sign = cv2.cvtColor(sign, cv2.COLOR_RGB2HSV)
        threshold_factor = 0.37
        threshold = hsv_sign[:, :, 2].max() * threshold_factor + hsv_sign[:, :, 2].min() * (1.0 - threshold_factor)
        mask = (hsv_sign[:, :, 2] < threshold).astype(np.uint8)
        mask_size = np.sum(mask)
        # compute contours to (try to) only extract the sign pictogram
        _, s_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # extract biggest area contour as the pictogram
        for c in s_contours:
            a = cv2.contourArea(c)
            if mask_size > a > 0.5 * mask_size:  # a < mask_size to prevent problem with sign bounding contours
                mask = cv2.drawContours(np.zeros_like(mask, dtype=np.uint8), [c], -1, 1, -1)
                mask_size = a
                break
        template_best_score = {}
        for name, t, flipped in template_generator(ref_sign_path):
            res = cv2.matchTemplate(mask, t, cv2.TM_CCORR)
            score = np.max(res)
            score /= max(mask_size, np.sum(t))
            if name not in template_best_score.keys() or template_best_score[name]['score'] < score:
                template_best_score[name] = {'file': name, 'flipped': flipped, 'score': score}
        matches = sorted(template_best_score.values(), key=lambda x: -x['score'])
        result_list.append({"sign": hsv_sign, "matches": matches})
    return result_list


def template_iterator(path: str):
    files_list = glob.glob(os.path.join(path, "*sng*.png"))
    size_factors = [0.8, 0.9, 0.95]
    for t in files_list:
        template = cv2.imread(t, cv2.IMREAD_UNCHANGED)
        template = template[:, :, 3] == 255
        for s in size_factors:
            resize_ratio = BOX_SIZE * s / max(template.shape)
            tem = cv2.resize(template.astype(np.uint8), (0, 0), fx=resize_ratio, fy=resize_ratio)
            yield t.split('_')[0], tem


def diamond_template_generator(sign_dir: str):
    for name, t in template_iterator(os.path.join(sign_dir, "diamond")):
        yield name, t, False
        yield name, np.flip(np.rot90(t), axis=1), True


def other_template_generator(sign_dir: str):
    for name, t in template_iterator(os.path.join(sign_dir, "others")):
        yield name, t, False
        yield name, np.flip(t, axis=1), True


def template_generator(sign_path):
    for name, t, flipped in diamond_template_generator(sign_path):
        yield name, t, flipped
    for name, t, flipped in other_template_generator(sign_path):
        yield name, t, flipped
