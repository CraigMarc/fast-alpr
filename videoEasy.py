 #this file detects license plates and records there number 
import cv2 as cv
from dataclasses import dataclass
import sys
import re
from statistics import mean
import numpy as np
import easyocr
from fast_alpr.alpr import ALPR, BaseOCR, OcrResult

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

resultsArr = []
checkArr = []

@dataclass
class Result:
    plate_number: str
    confidence: float
    video_time: float
    file_name: str



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
       
        # Convert to format compatible with EasyOCR (not needed fast alrp already processes)
        #plate_image = cv.cvtColor(cropped_plate, cv.COLOR_BGR2RGB)
       

        # Use EasyOCR to read text from plate
        plate_number = reader.readtext(cropped_plate)
        
        concat_number = ' '.join([number[1] for number in plate_number])
        number_conf = np.mean([number[2] for number in plate_number])
        print(concat_number)
        print(number_conf)
        return OcrResult(text=concat_number, confidence=number_conf)


alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr=EzOCR())
""""
alpr_results = alpr.predict("assets/test_image.png")
print(alpr_results)
"""

# You can also initialize the ALPR with custom plate detection and OCR models.
"""
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)
"""
### get to work with video files

# Open the video file (replace with your video file path)
video_path = file_name
cap = cv.VideoCapture(video_path)
video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
video_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

# Frame skipping factor (adjust as needed for performance)
frame_skip = 3  # Skip every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no frames left

    # Skip frames
    if frame_skip != 0 and frame_count % frame_skip != 0:
        frame_count += 1
        continue  # Skip processing this frame

    frame_count += 1

    # Resize the frame (optional, adjust size as needed)
    frame = cv.resize(frame, (1366, 768))  # Resize to 640x480
     #frame = cv.resize(frame, (video_width, video_height))  # auto resize ******************

    # Make predictions on the current frame
    #results = model.predict(source=frame)
    alpr_results = alpr.predict(frame)
    
    #print(alpr_results)
    if len(alpr_results) !=0:
        timeElapsed = round(cap.get(cv.CAP_PROP_POS_MSEC)/1000, 2)
        print(alpr_results[0].ocr.text, alpr_results[0].ocr.confidence)
        print(timeElapsed)
        
        # put results into object
        
        if alpr_results[0].ocr.text not in checkArr and alpr_results[0].ocr.confidence > 0.8:
           resultsArr.append(
                Result(
                    plate_number=alpr_results[0].ocr.text,
                    confidence=round(alpr_results[0].ocr.confidence, 3),
                    video_time=timeElapsed,
                    file_name=file_name
                )
                )

        checkArr.append(alpr_results[0].ocr.text)

    # Show the frame with detections (show while video progresses)
    """
    cv.imshow('Detections', frame)

    # Write the frame to the output video (optional)
    out.write(frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

    frame_count += 1  # Increment frame count
    """
# print results list
    
print(resultsArr)

# Release resources

cap.release()
#out.release()  # Release the VideoWriter object if used
cv.destroyAllWindows()
