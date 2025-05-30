# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

import cv2 as cv
from fast_alpr import ALPR

# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# The "assets/test_image.png" can be found in repo root dit
# You can also pass a NumPy array containing cropped plate image
alpr_results = alpr.predict("./test2.jpg")
print(alpr_results)
