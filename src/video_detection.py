import cv2
import os
import argparse
from src.detector.detection_pipeline import detect_signs
import cProfile
import pstats
import io
import numpy as np


def main(video_path):
    pr = cProfile.Profile()
    cap = cv2.VideoCapture(video_path)

    pr.enable()
    while True:
        # Capture frame-by-frame
        read_ok, frame = cap.read()

        if not read_ok:
            break

        frame = cv2.resize(frame, (1224, 1024))
        hsv_im = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # contours = detect_signs(hsv_im)

        # layer = np.zeros_like(hsv_im)
        # layer[:, :, 1] = hsv_im[:, :, 1]

        # im_bounding_boxes = cv2.drawContours(layer,
        #                                      contours,
        #                                      -1,
        #                                      (255, 0, 0),
        #                                      3)

        # Display the resulting frame
        cv2.imshow('frame', hsv_im[:, :, 2])
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pr.disable()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-video',
                        required=True,
                        type=str,
                        dest="path")
    args = parser.parse_args()

    main(args.path)