import cv2
from fast_alpr import ALPR

# Initialize the ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# Load the image
image_path = "data/test.png"
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (1366, 768))  # Resize to 640x480
# Draw predictions on the image
annotated_frame = alpr.draw_predictions(frame)

# Display the result
cv2.imshow("ALPR Result", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()