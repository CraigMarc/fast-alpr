# this file detects license plates and records there number 
import cv2 as cv
from fast_alpr import ALPR
import sys
import re
from statistics import mean
import numpy as np
import easyocr
from fast_alpr.alpr import ALPR, BaseOCR, OcrResult

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)


#get file name from command line

if len(sys.argv) > 1:
    file_name = sys.argv[1]
else: 
    file_name = 'data/dash1.mp4'

# setup EzOCR

class EzOCR(BaseOCR):
    def __init__(self) -> None:
        
       """
        Init EasyOCR.
        """

    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        if cropped_plate is None:
            return None
        # You can change 'eng' to the appropriate language code as needed
      
        # Convert to format compatible with EasyOCR

        # fast-alrp seems to already do the image processing processing futher makes more inaccurate
        """"
        gray = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv.filter2D(gray, -1, sharpen_kernel)
        thresh = cv.threshold(sharpen, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        plate_image = thresh
        plate_array = np.array(plate_image)"""

        # Use EasyOCR to read text from plate
        plate_number = reader.readtext(cropped_plate)
        
        concat_number = ' '.join([number[1] for number in plate_number])
        number_conf = np.mean([number[2] for number in plate_number])
        print(concat_number)
        print(number_conf)
        return OcrResult(text=concat_number, confidence=number_conf)





# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr=EzOCR())

### get to work with video files

# Open the video file (replace with your video file path)
video_path = file_name
cap = cv.VideoCapture(video_path)
video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
video_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

# Create a VideoWriter object (optional, if you want to save the output)

output_path = 'output_videoEz.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, 30.0, (1366, 768))  # Adjust frame size if necessary




# Frame skipping factor (adjust as needed for performance)
frame_skip = 3  # Skip every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no frames left

    # Skip frames remove if want to skip frames ******
    
    if frame_skip != 0 and frame_count % frame_skip != 0:
        frame_count += 1
        continue  # Skip processing this frame

    frame_skip += 1
        
    # Resize the frame (optional, adjust size as needed)
    frame = cv.resize(frame, (1366, 768))  # Resize to 640x480
     #frame = cv.resize(frame, (video_width, video_height))  # auto resize ******************

    # Draw predictions on the image
    annotated_frame = alpr.draw_predictions(frame)   
            
            
    # Show the frame with detections (show while video progresses)
    
    cv.imshow('Detections', frame)

    # Write the frame to the output video (optional)
    out.write(frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

    frame_count += 1  # Increment frame count
    
# Release resources

cap.release()
out.release()  # Release the VideoWriter object if used
cv.destroyAllWindows()