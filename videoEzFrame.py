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
        gray = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv.filter2D(gray, -1, sharpen_kernel)
        thresh = cv.threshold(sharpen, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        plate_image = thresh
        plate_array = np.array(plate_image)

        # Use EasyOCR to read text from plate
        plate_number = reader.readtext(plate_image)
        
        concat_number = ' '.join([number[1] for number in plate_number])
        number_conf = np.mean([number[2] for number in plate_number])
        print(concat_number)
        print(number_conf)
        return OcrResult(text=concat_number, confidence=number_conf)


alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr=EzOCR())


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

# Create a VideoWriter object (optional, if you want to save the output)

output_path = 'output_video.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, 30.0, (640, 480))  # Adjust frame size if necessary




# Frame skipping factor (adjust as needed for performance)
frame_skip = 3  # Skip every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no frames left

    # Skip frames
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue  # Skip processing this frame

    # Resize the frame (optional, adjust size as needed)
    frame = cv.resize(frame, (640, 480))  # Resize to 640x480
    

    # Draw predictions on the image
    annotated_frame = alpr.draw_predictions(frame)   

    # Make predictions on the current frame
    #results = model.predict(source=frame)
    #alpr_results = alpr.predict(frame)
    """
    #print(alpr_results)
    if len(alpr_results) !=0:
        print(alpr_results[0].ocr.text, alpr_results[0].ocr.confidence)
        print(str(cap.get(cv.CAP_PROP_POS_MSEC)))
        #print(alpr_results[0].detection.bounding_box)
       

    # Iterate over results and draw predictions
    
    for result in alpr_results:
        boxes = result.boxes  # Get the boxes predicted by the model
        for box in boxes:
            class_id = int(box.cls)  # Get the class ID
            confidence = box.conf.item()  # Get confidence score
            coordinates = box.xyxy[0]  # Get box coordinates as a tensor

            # Extract and convert box coordinates to integers
            x1, y1, x2, y2 = map(int, coordinates.tolist())  # Convert tensor to list and then to int

            # Draw the box on the frame
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            
            
            
            
            # Try to apply OCR on detected region
            
            try:
                # Ensure coordinates are within frame bounds
                r0 = max(0, x1)
                r1 = max(0, y1)
                r2 = min(frame.shape[1], x2)
                r3 = min(frame.shape[0], y2)

                # Crop license plate region
                plate_region = frame[r1:r3, r0:r2]

                # Convert to format compatible with EasyOCR
                plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
                plate_array = np.array(plate_image)

                # Use EasyOCR to read text from plate
                plate_number = reader.readtext(plate_array)
                concat_number = ' '.join([number[1] for number in plate_number])
                number_conf = np.mean([number[2] for number in plate_number])

                # Draw the detected text on the frame
                cv.putText(
                    img=frame,
                    text=f"Plate: {concat_number} ({number_conf:.2f})",
                    org=(r0, r1 - 10),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255),
                    thickness=2
                )

            except Exception as e:
                print(f"OCR Error: {e}")
                pass
                """
         
            
            
    # Show the frame with detections (show while video progresses)
    
    cv.imshow('Detections', frame)

    # Write the frame to the output video (optional)
    out.write(frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

    frame_count += 1  # Increment frame count
    
# Release resources

cap.release()
#out.release()  # Release the VideoWriter object if used
cv.destroyAllWindows()
