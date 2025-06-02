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

# Load the image
image_path = "data/test.png"
frame = cv.imread(image_path)
frame = cv.resize(frame, (1366, 768))  # Resize to 640x480
# Draw predictions on the image
annotated_frame = alpr.draw_predictions(frame)

# Display the result
cv.imshow("ALPR Result", annotated_frame)
cv.waitKey(0)
cv.destroyAllWindows()