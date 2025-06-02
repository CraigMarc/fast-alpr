import cv2 as cv
from fast_alpr import ALPR
import sys
import re
from statistics import mean
import numpy as np
import easyocr
from fast_alpr.alpr import ALPR, BaseOCR, OcrResult
from PIL import Image

import cv2 as cv
from fast_alpr import ALPR

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# You can also initialize the ALPR with custom plate detection and OCR models.
"""
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)
"""


# The "assets/test_image.png" can be found in repo root dit
# You can also pass a NumPy array containing cropped plate image


 # Draw the predicted bounding box


class ez(BaseOCR):
    def __init__(self) -> None:
        
        """
        Init PytesseractOCR.
        """
        

    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        if cropped_plate is None:
            return None
        # You can change 'eng' to the appropriate language code as needed
        plate_number = reader.readtext(cropped_plate)
        
        concat_number = ' '.join([number[1] for number in plate_number])
        number_conf = np.mean([number[2] for number in plate_number])
        print(concat_number)
        print(number_conf)
        return OcrResult(text=concat_number, confidence=number_conf)
           



alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr=ez())
alpr_results = alpr.predict("data/test.png")

print(alpr_results)
print(alpr_results[0].ocr.text)
#print(vars(alpr_results[0]))
