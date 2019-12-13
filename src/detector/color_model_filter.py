import cv2
import numpy as np

# globally define kernels (do not recreate them for each image)
small_closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
# erode_kernel = np.ones((3, 3), dtype=np.uint8)
# dilating_kernel = np.ones((15,15), dtype=np.uint8)

# CLAHE = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))


def generate_mask_from_color_model(hsv_image: np.ndarray, fallback_to_percentile_threshold: bool=True):
    mask = generate_raw_mask(hsv_image, percentile_fallback=fallback_to_percentile_threshold)
    mask = filter_mask_noise(mask)
    return mask


def generate_raw_mask(hsv_image: np.ndarray, percentile_fallback: bool=True):
    hue = cv2.GaussianBlur(hsv_image[:, :, 0], (3, 3), 0)
    hue_mask = (hue < 40) & (hue > 10)
    yellow_saturation = cv2.GaussianBlur(hsv_image[:, :, 1], (3, 3), 0)
    yellow_saturation = yellow_saturation * hue_mask
    # define saturation threshold using value histogram
    smin, smax = np.percentile(yellow_saturation, [97, 99.87])
    hist, col = np.histogram(yellow_saturation, 25)
    correct_threshold = int(hue_mask.sum() * 0.025)
    last_v = correct_threshold
    last_convex_index = None
    was_increasing = False
    for i, v in enumerate(hist):
        if i < 8:
            continue
        if col[i] > smax > 100:
            break
        # if variation goes from decreasing to increasing keep index
        if v >= last_v and hist[i] < correct_threshold:
            if not was_increasing:
                last_convex_index = i
            # print(last_convex_index, v, last_v)
            was_increasing = True
        elif v < 0.95 * last_v:
            was_increasing = False
        last_v = v

    # if no curve inversion found fall back to the old fashioned percentile method
    if last_convex_index is None:
        if percentile_fallback:
            threshold = int(0.6 * (smax - smin) + smin)
        else:
            raise HistogramThresholdDetectionFailedError
    else:
        if col[last_convex_index - 2] > smin:
            last_convex_index -= 3
        else:
            last_convex_index -= 1
        threshold = col[last_convex_index]
    # print("elected threshold:", threshold)
    return (yellow_saturation > threshold).astype(np.uint8)


def filter_mask_noise(mask: np.ndarray):
    # Close sign to fix imperfect color model
    mask = cv2.dilate(mask, small_closing_kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_closing_kernel)  # close sign
    # Fill contours to prevent problem of splited sign when removing noise
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, 1, -1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_closing_kernel)  # close sign
    # mask = cv2.erode(mask, small_closing_kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_kernel)  # remove noise
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)  # close sign
    return mask


class HistogramThresholdDetectionFailedError(Exception):
    pass

