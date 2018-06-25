import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import cProfile, pstats, io

from src.detector.detection_pipeline import detect_and_extract_signs
from src.classifier.pattern_matching import match_sign_with_pattern
from src.utils.sign_creator import speed_sign_creator


def main():
    pr = cProfile.Profile()
    file = "data/{:03d}.jpg".format(random.randint(1, 404))
    # file = "data/{:03d}.jpg".format(322)
    print(file)
    im = cv2.imread(file)
    im = im[30:, :, :]
    im = cv2.resize(im, (1224, 1024))
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # pr.enable()
    sign_images = detect_and_extract_signs(hsv_im)
    # pr.disable()
    pr.enable()
    matches_list = match_sign_with_pattern(sign_images, 'sign')
    pr.disable()
    plt.figure(0)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.figure(1)
    for s, matches_dict in enumerate(matches_list):
        plt.subplot(len(matches_list), 4, 1 + 4 * s)
        plt.imshow(cv2.cvtColor(matches_dict['sign'], cv2.COLOR_HSV2RGB))
        plt.title("Extracted sign")
        for i, match in enumerate(matches_dict['matches'][:3]):
            if 'speed_limit' in match.keys():
                ref_sign = speed_sign_creator(speed=match['speed_limit'])
                ref_sign = cv2.cvtColor(ref_sign, cv2.COLOR_BGR2RGB)
            else:
                sign_file = "{}_out-01.png".format(match['file'])
                ref_sign = cv2.cvtColor(cv2.imread(sign_file), cv2.COLOR_BGR2RGB)
                if match['flipped']:
                    ref_sign = np.flip(ref_sign, axis=1)
            plt.subplot(len(matches_list), 4, 2 + i + 4 * s)
            plt.imshow(ref_sign)
            plt.title("score: {:.3f}".format(match['score']))
            plt.tick_params(which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            left=False,  # ticks along the top edge are off
                            labelbottom=False,  # labels along the bottom edge are off
                            labelleft=False)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
    plt.show()


if __name__ == '__main__':
    main()

