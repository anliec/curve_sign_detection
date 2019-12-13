import numpy as np
import cv2

from src.detector.color_model_filter import generate_mask_from_color_model
from src.detector.color_model_filter import HistogramThresholdDetectionFailedError
from src.detector.sign_contour_detection import find_sign_contour
from src.detector.sign_extractor import extract_sign_image

MASKING_ERODE_KERNEL = np.ones((30, 30))


def detect_and_extract_signs(hsv_image: np.ndarray):
    sign_contours = detect_signs(hsv_image)
    return extract_sign_image(sign_contours, hsv_image)


def detect_signs(hsv_image: np.ndarray):
    """
        Detect signs from the given image
        :param hsv_image: cv2 image in the HSV color format
        :return: a list of contour of signs
        """
    masked_image = hsv_image.copy()
    mask = generate_mask_from_color_model(masked_image)
    first_mask_size = np.count_nonzero(mask)
    sign_contours = []
    while True:
        new_sign_contours = find_sign_contour(mask)
        if len(new_sign_contours) == 0:
            break
        sign_contours += new_sign_contours
        # mask sign on image and rerun detection (try to find a second sign with lower saturation)
        old_sign_mask = cv2.erode(cv2.drawContours(np.ones_like(mask), sign_contours, -1, 0, -1), MASKING_ERODE_KERNEL)
        masked_image = cv2.bitwise_and(masked_image, masked_image, mask=old_sign_mask)
        # cv2.imwrite("masked_image.png", masked_image)
        try:
            mask = generate_mask_from_color_model(masked_image, fallback_to_percentile_threshold=False)
        except HistogramThresholdDetectionFailedError:
            break  # histogram threshold failed to find an other possible sign, no need to continue
        # if the new mask is too big, it's probably just noise, no need to continue
        mask_size = np.count_nonzero(mask)
        # print(mask_size, first_mask_size, 6 * first_mask_size)
        if mask_size > 6 * first_mask_size:
            break
    return sign_contours
