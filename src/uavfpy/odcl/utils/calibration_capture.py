"""capture an image for calibration.
"""

import cv2
import os
import argparse

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--o", required=True, help="output_folder")
    opt = arg.parse_args()
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()

    print("Writing to chessboard file")
    files = os.listdir(opt.o)
    index = 0
    while f"img{index}.jpg" in files:
        index += 1
    cv2.imwrite(f"{opt.o}/img{index}.jpg", img)
