import cv2
import numpy as np
import glob
import os
import itertools
import pytesseract
import re

from src.detector.sign_extractor import BOX_SIZE

MATCHING_SIGN_SIZE = 100
SPEED_REGEX = re.compile("[0-9][0-9]", re.MULTILINE)


def match_sign_with_pattern(sign_image_list, ref_sign_path: str="sign"):
    return list(map(match_sign, zip(sign_image_list, itertools.repeat(ref_sign_path))))


def match_sign(arg):
    hsv_sign, ref_sign_path = arg
    threshold_factor = 0.8
    v_max, v_min = np.percentile(hsv_sign[:, :, 2], [20, 60])
    threshold = v_max * threshold_factor + v_min * (1.0 - threshold_factor)
    mask = (hsv_sign[:, :, 2] < threshold).astype(np.uint8)
    # compute contours to (try to) only extract the sign pictogram
    _, s_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # remove area with no point near the center
    threshold = BOX_SIZE // 5
    for c in s_contours:
        c_centered_coord = (c > threshold) * (c < (BOX_SIZE - threshold))
        if np.sum(np.multiply(c_centered_coord[:, :, 0], c_centered_coord[:, :, 1])) == 0:
            # if no border point is in the center, then remove the area
            if cv2.contourArea(c) > 0.75 * BOX_SIZE ** 2:
                # if th area cover the entire image, we probably have the sign boundary,
                # here we just remove some pixels around he contour
                mask = cv2.drawContours(mask, [c], -1, 0, 5)
            else:
                mask = cv2.drawContours(mask, [c], -1, 0, -1)
    small_mask = cv2.resize(mask, (MATCHING_SIGN_SIZE, MATCHING_SIGN_SIZE))
    mask_size = np.sum(small_mask)

    template_best_score = {}
    for name, t, flipped in template_generator(ref_sign_path):
        res = cv2.matchTemplate(small_mask, t, cv2.TM_CCORR)
        score = np.max(res)
        score /= max(mask_size, np.sum(t))
        if name not in template_best_score.keys() or template_best_score[name]['score'] < score:
            template_best_score[name] = {'file': name, 'flipped': flipped, 'score': score}
    matches = sorted(template_best_score.values(), key=lambda x: -x['score'])
    if matches[0]['score'] < 0.6:
        # OCR
        ocr_img = np.ones((BOX_SIZE * 2, BOX_SIZE * 2), dtype=np.uint8) * 255
        ocr_img[BOX_SIZE // 2:BOX_SIZE + BOX_SIZE // 2, BOX_SIZE // 2:BOX_SIZE + BOX_SIZE // 2] = 255 - mask * 255
        ocr_img = cv2.putText(ocr_img, "Sign", (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 2)
        text = pytesseract.image_to_string(ocr_img, lang='eng').replace('\n', ' ')
        regex_match = SPEED_REGEX.search(text)
        if regex_match:
            matches.insert(0, {'score': 1.0, 'speed_limit': int(regex_match.group()), 'flipped': False})
    return {"sign": hsv_sign, "matches": matches}


def template_iterator(path: str):
    files_list = glob.glob(os.path.join(path, "*sng*.png"))
    size_factors = [0.85, 0.95]
    for t in files_list:
        template = cv2.imread(t, cv2.IMREAD_UNCHANGED)
        template = template[:, :, 3] == 255
        size_ratio = MATCHING_SIGN_SIZE / max(template.shape)
        for s in size_factors:
            resize_ratio = s * size_ratio
            tem = cv2.resize(template.astype(np.uint8), (0, 0), fx=resize_ratio, fy=resize_ratio)
            yield t.split('_')[0], tem


def diamond_template_generator(sign_dir: str):
    for name, t in template_iterator(os.path.join(sign_dir, "diamond")):
        yield name, t, False
        yield name, np.flip(np.rot90(t), axis=1), True


def other_template_generator(sign_dir: str):
    for name, t in template_iterator(os.path.join(sign_dir, "others")):
        # resize to a square as the sign are transformed that way
        s = max(t.shape)
        t = cv2.resize(t, (s, s))
        yield name, t, False
        yield name, np.flip(t, axis=1), True


def template_generator(sign_path):
    for name, t, flipped in diamond_template_generator(sign_path):
        yield name, t, flipped
    for name, t, flipped in other_template_generator(sign_path):
        yield name, t, flipped
