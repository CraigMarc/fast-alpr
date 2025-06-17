# this file detects license plates and records there number 
import cv2 as cv
from fast_alpr import ALPR
from dataclasses import dataclass
import sys
import csv
import os
import time

resultsArr = []
checkArr = []
imgArr= []


#get file name from command line

if len(sys.argv) > 1:
    file_name = sys.argv[1]
else: 
    file_name = 'data/real.avi'

# get video creation time

ti_m = os.path.getmtime(file_name)

# Converting the time in seconds to a timestamp

creation_time = time.ctime(ti_m)



# You can also initialize the ALPR with custom plate detection and OCR models.
#best model
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)
"""
# argentinian plate model
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="argentinian-plates-cnn-model",
)

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="argentinian-plates-cnn-synth-model",
)
# 2nd best model
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="european-plates-mobile-vit-v2-model",
)"""


### get to work with video files

# Open the video file (replace with your video file path)
video_path = file_name
cap = cv.VideoCapture(video_path)
video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
print(video_height)
print(video_width)

# Frame skipping factor (adjust as needed for performance)
frame_skip = 3  # Skip every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no frames left
   
    # Skip frames
    if frame_skip != 0 and frame_count % frame_skip != 0:
        frame_count = frame_count + 1
       
        continue  # Skip processing this frame

    frame_count = frame_count + 1
  

    # Resize the frame (optional, adjust size as needed)
    #frame = cv.resize(frame, (1366, 768))  # Resize to improve accuracy *****************
    frame = cv.resize(frame, (video_width, video_height))  # auto resize ******************
    

    # Make predictions on the current frame
    #results = model.predict(source=frame)
    alpr_results = alpr.predict(frame)
    
    #print(alpr_results)
    if len(alpr_results) !=0:
        timeElapsed = round(cap.get(cv.CAP_PROP_POS_MSEC)/1000, 2)
        print(alpr_results[0].ocr.text, alpr_results[0].ocr.confidence)
        print(timeElapsed)
        
       
        # loop through results and add to the dictionary
        
        for x in alpr_results:
            
            # save the image with the highest confidence for each plate
            for iter in imgArr:
                if iter["plate_number"] == x.ocr.text and iter["confidence"] < x.ocr.confidence:
                
                    iter["confidence"] = x.ocr.confidence
                    jpg_filenameNew = "jpeg/" + x.ocr.text  + ".jpg"
                    cv.imwrite(jpg_filenameNew, frame)     # save frame as JPEG file
           

            if x.ocr.text not in checkArr and x.ocr.confidence >= 0.97:
                # save image fram to file later only save the image and data with the highest confidence ********
                #jpg_filename = "jpeg/" + filename[:-4] + str(frame_count) + ".jpg"
                jpg_filename = "jpeg/" + x.ocr.text + "_c" + str(int(x.ocr.confidence * 100000)) + ".jpg"
                cv.imwrite(jpg_filename, frame)     # save frame as JPEG file
                data = {
                "plate_number": x.ocr.text,
                "confidence": round(x.ocr.confidence, 3),
                "video_time": timeElapsed,
                "file_name": file_name,
                "creation_time": creation_time
                }
                resultsArr.append(data)

                chkData = {
                "plate_number": x.ocr.text,
                "confidence": x.ocr.confidence
                }
        
                imgArr.append(chkData)
                checkArr.append(x.ocr.text)

        
        
        # one result per frame
        """
        if alpr_results[0].ocr.text not in checkArr and alpr_results[0].ocr.confidence > 0.97:
          
            data = {
            "plate_number": alpr_results[0].ocr.text,
            "confidence": round(alpr_results[0].ocr.confidence, 3),
            "video_time": timeElapsed,
            "file_name": file_name,
            "creation_time": creation_time
            }
                
            resultsArr.append(data)

            checkArr.append(alpr_results[0].ocr.text)"""

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

#save data to a csv file
# change a to w to start new file a to append
with open('spreadsheet.csv', 'a', newline='') as csvfile:
    fieldnames = ['plate_number', 'confidence', 'video_time', 'file_name', 'creation_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #unquote for headers to be written
    #writer.writeheader()
    writer.writerows(resultsArr)
    

# Release resources

cap.release()
#out.release()  # Release the VideoWriter object if used
cv.destroyAllWindows()
