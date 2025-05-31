import cv2 as cv
from fast_alpr import ALPR

# You can also initialize the ALPR with custom plate detection and OCR models.

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)



# The "assets/test_image.png" can be found in repo root dit
# You can also pass a NumPy array containing cropped plate image
alpr_results = alpr.predict("data/test2.jpg")

print(alpr_results[0].ocr.text)
#print(vars(alpr_results[0]))
