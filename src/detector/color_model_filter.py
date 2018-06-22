import cv2
import numpy as np

# globally define kernels (do not recreate them for each image)
small_closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
# erode_kernel = np.ones((3, 3), dtype=np.uint8)
# dilating_kernel = np.ones((15,15), dtype=np.uint8)

# CLAHE = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))


def generate_mask_from_color_model(hsv_image: np.ndarray):
    mask = generate_raw_mask(hsv_image)
    mask = filter_mask_noise(mask)
    return mask


def generate_raw_mask(hsv_image: np.ndarray):
    hue = cv2.GaussianBlur(hsv_image[:, :, 0], (3, 3), 0)
    # hue = hsv_image[:, :, 0]
    hue_mask = (hue < 35) & (hue > 10)
    yellow_saturation = hsv_image[:, :, 1] * hue_mask
    # non_masked_val = yellow_saturation[hue_mask]
    # smax = non_masked_val.max()
    # smin = non_masked_val.min()
    yellow_saturation = cv2.GaussianBlur(yellow_saturation, (3, 3), 0)
    # yellow_saturation = CLAHE.apply(yellow_saturation)
    smin, smax = np.percentile(yellow_saturation, [90, 95])
    threshold = int(0.65 * (smax - smin)) + smin
    print(smin, smax, threshold)
    augmented_saturation_mask = (yellow_saturation > threshold).astype(np.uint8)
    return augmented_saturation_mask


def filter_mask_noise(mask: np.ndarray):
    # Close sign to fix imperfect color model
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_closing_kernel)  # close sign
    # Fill contours to prevent problem of splited sign when removing noise
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, 1, -1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_closing_kernel)  # close sign
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_kernel)  # remove noise
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)  # close sign
    return mask




