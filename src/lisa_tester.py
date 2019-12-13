import sqlite3 as sql
import cv2
import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from src.detector.detection_pipeline import detect_and_extract_signs
from src.classifier.pattern_matching import match_sign_with_pattern

EVALUATED_SIGNS = {'w1-1n': 'turnLeft',
                   'w1-1m': 'turnRight',
                   'w2-1n': 'intersection',
                   'w1-2m': 'curveRight',
                   'w1-2n': 'curveLeft',
                   'w4-3n': 'addedLane',
                   'w4-1n': 'merge'}
BOUNDING_BOX_TOLERANCE = 10


def main(dataset_path: str):
    all_annotation_df = pd.read_csv(os.path.join(dataset_path, "allAnnotations.csv"), sep=';')
    con = sql.connect(":memory:")
    all_annotation_df.to_sql("annotation", con)

    detection_true_positive = 0
    detection_false_positive = 0
    # detection_true_negative = 0
    detection_false_negative = 0
    count = 0

    sql_query = "SELECT * FROM annotation WHERE Annotation_tag IN {} ".format(tuple(EVALUATED_SIGNS.values()))
    sql_query += "AND Filename LIKE 'vid%'"
    for row in con.execute(sql_query):
        file_path = row[1]
        sign_class = row[2]

        im = cv2.imread(os.path.join(dataset_path, file_path))
        base_size = im.shape
        im = cv2.resize(im, (1224, 1024))

        up_left_point = (int((int(row[3]) - BOUNDING_BOX_TOLERANCE) * 1224 / base_size[1]),
                         int((int(row[4]) - BOUNDING_BOX_TOLERANCE) * 1024 / base_size[0]))
        bot_right_point = (int((int(row[5]) + BOUNDING_BOX_TOLERANCE) * 1224 / base_size[1]),
                           int((int(row[6]) + BOUNDING_BOX_TOLERANCE) * 1024 / base_size[0]))

        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        sign_images = detect_and_extract_signs(hsv_im)
        matches_list = match_sign_with_pattern(sign_images, 'sign')

        detected = False
        for match in matches_list:
            contour = match['bounding_box']
            # print("contour", contour)
            # max_point = np.max(contour, axis=0)[0]
            # min_point = np.min(contour, axis=0)[0]
            avg_point = np.average(contour, axis=0)[0]
            if (avg_point < bot_right_point).all() and (avg_point > up_left_point).all():
                detection_true_positive += 1
                detected = True
            else:
                detection_false_positive += 1
        if not detected:
            detection_false_negative += 1
        count += 1

        if detected:
            print(file_path)
            plt.figure(0)
            im_bounding_boxes = cv2.drawContours(im.copy(),
                                                 list(map(lambda x: x['bounding_box'], matches_list)),
                                                 -1,
                                                 (0, 255, 0),
                                                 3)
            im_bounding_boxes = cv2.rectangle(im_bounding_boxes, up_left_point, bot_right_point, (255, 0, 255), 3)
            plt.imshow(cv2.cvtColor(im_bounding_boxes, cv2.COLOR_BGR2RGB))
            plt.show()
            # break

    print("true positive:", detection_true_positive)
    print("false positive:", detection_false_positive)
    print("false negative:", detection_false_negative)
    print("count:", count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        type=str,
                        dest="path")
    args = parser.parse_args()

    main(args.path)


