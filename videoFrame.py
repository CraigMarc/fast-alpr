# this file detects license plates and records there number 
import cv2 as cv
from fast_alpr import ALPR
import sys

#get file name from command line

if len(sys.argv) > 1:
    file_name = sys.argv[1]
else: 
    file_name = 'data/dash1.mp4'


# You can also initialize the ALPR with custom plate detection and OCR models.

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

### get to work with video files

# Open the video file (replace with your video file path)
video_path = file_name
cap = cv.VideoCapture(video_path)

# Create a VideoWriter object (optional, if you want to save the output)

output_path = 'output_video.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, 30.0, (1366, 768))  # Adjust frame size if necessary

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
    frame = cv.resize(frame, (1366, 768))  # Resize to 640x480
    

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
