import cv2 as cv
from fast_alpr import ALPR
import sys
import csv
import os
import time

#get file name from command line

if len(sys.argv) > 1:
    directory = sys.argv[1]
else: 
    directory = r"C:\Users\Criag\Videos\dashcam_data\testbatch\DCIM"


# You can also initialize the ALPR with custom plate detection and OCR models.
#best model
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

#function to analyze video

def analyze_video (whole_path, resultsArr, checkArr, filename, creation_time):
    ### get to work with video files

        # Open the video file (replace with your video file path)
        video_path = whole_path
        cap = cv.VideoCapture(video_path)
        video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        # Frame skipping factor (adjust as needed for performance)
        frame_skip = 0 # Skip every 3rd frame
        frame_count = 0
        timeCount = 0

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
            #frame = cv.resize(frame, (1280, 720))  # Resize to improve accuracy *****************
            frame = cv.resize(frame, (video_width, video_height))  # auto resize ******************

            # Make predictions on the current frame
            #results = model.predict(source=frame)
            alpr_results = alpr.predict(frame)
            timeElapsed = round(cap.get(cv.CAP_PROP_POS_MSEC)/1000, 2)
            timeCount = timeCount + 1

            if timeCount == 0:
                timeCount = 0
                print("analyzing:" + " " + whole_path + " " + str(timeElapsed) + "sec")
            
            #print(alpr_results)
            if len(alpr_results) !=0:
                
                print(alpr_results[0].ocr.text, alpr_results[0].ocr.confidence)
                print(timeElapsed)
                
            
                # loop through results and add to the dictionary
                
                for x in alpr_results:
                    if x.ocr.text not in checkArr and x.ocr.confidence >= 0.97:
                       
                        data = {
                        "plate_number": x.ocr.text,
                        "confidence": round(x.ocr.confidence, 3),
                        "video_time": timeElapsed,
                        "file_name": filename,
                        "creation_time": creation_time
                        }
                        resultsArr.append(data)
                
                        checkArr.append(x.ocr.text)
                

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

        print("results:")
        print(resultsArr)

        #save data to a csv file
        # change a to w to start new file a to append to end of current file
        with open('spreadsheet_batchf.csv', 'a', newline='') as csvfile:
            fieldnames = ['plate_number', 'confidence', 'video_time', 'file_name', 'creation_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #unquote for headers to be written
            #writer.writeheader()
            writer.writerows(resultsArr)
            # Release resources

        cap.release()
        #out.release()  # Release the VideoWriter object if used
        cv.destroyAllWindows()
            

# function to get subdirectories

def get_subdirectories():
    sub_directories = []

    for file in os.listdir(directory):
        pathname = os.fsdecode(file)
        sub_directories.append(pathname)
    return sub_directories


# get files from subdirectories and analyze

def get_files():

    sub = (get_subdirectories())

    for val in sub:
        directory_2 = directory + "\\" + val
        
        for file in os.listdir(directory_2):
            filename = os.fsdecode(file)
            whole_path = directory_2 + "\\" + filename
            print(whole_path)

            resultsArr = []
            checkArr = []

            # get creation time of video file

            ti_m = os.path.getmtime(whole_path)

            # Converting the time in seconds to a timestamp

            creation_time = time.ctime(ti_m)

            #call analysis function for each file
            analyze_video(whole_path, resultsArr, checkArr, filename, creation_time)

# call get files to start analysis
get_files()
       





        

