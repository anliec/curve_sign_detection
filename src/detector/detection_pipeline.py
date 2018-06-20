import numpy as np

from src.detector.color_model_filter import generate_mask_from_color_model
from src.detector.sign_contour_detection import find_sign_contour
from src.detector.sign_extractor import extract_sign_image


def detect_and_extract_signs(hsv_image: np.ndarray):
    """
    Detect and extract sign images from the given image
    :param hsv_image: cv2 image in the HSV color format
    :return: a list of sign image in the HSV color format
    """
    mask = generate_mask_from_color_model(hsv_image)
    sign_contours = find_sign_contour(mask)
    return extract_sign_image(sign_contours, hsv_image)

