#video batchf same as video batch expect the code is split up into more functions can eventually delete video batch

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
    directory = r"C:\fast_alpr\dashcam_data\testbatch2\DCIM"


# You can also initialize the ALPR with custom plate detection and OCR models.
#best model
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

def get_creation_time (whole_path):
    # get creation time of video file

        ti_m = os.path.getmtime(whole_path)

        # Converting the time in seconds to a timestamp

        creation_time = time.ctime(ti_m)
        return creation_time
       
# function to add new plate

def add_new_plate (x, checkArr, filename, frame, timeElapsed, whole_path, resultsArr, imgArr):

    creation_time = get_creation_time(whole_path)
   
    if x.ocr.text not in checkArr and x.ocr.confidence >= 0.95:
        # other files may save later ???????????????????   
        #jpg_filename = "jpeg/" + filename[:-4] + str(frame_count) + ".jpg"
        #jpg_filename = "C:/Users/Criag/Videos/jpeg_files/" + x.ocr.text + "_c" + str(int(x.ocr.confidence * 100000)) + "_fn" + filename + ".jpg"
        #cv.imwrite(jpg_filename, frame)     # save frame as JPEG file 
        

        # save first plate image to jpeg_best folder

        #create folder if it does not exist

        if not os.path.exists("C:/fast_alpr/jpeg_best/"):
            os.makedirs("C:/fast_alpr/jpeg_best/")

        jpg_filenameNew = "C:/fast_alpr/jpeg_best/" + x.ocr.text + "_fn" + filename + ".jpg"
        cv.imwrite(jpg_filenameNew, frame)     # save frame as JPEG file
        data = {
        "plate_number": x.ocr.text,
        "confidence": round(x.ocr.confidence, 3),
        "video_time": timeElapsed,
        "file_name": whole_path,
        "creation_time": creation_time
        }
        resultsArr.append(data)

        chkData = {
        "plate_number": x.ocr.text,
        "confidence": x.ocr.confidence
        }
        
        imgArr.append(chkData)
            
        checkArr.append(x.ocr.text)

def best_image (filename, frame, imgArr, x):
   
    for iter in imgArr:
        if iter["plate_number"] == x.ocr.text and iter["confidence"] < x.ocr.confidence:

            iter["confidence"] = x.ocr.confidence
            jpg_filenameNew = "C:/fast_alpr/jpeg_best/" + x.ocr.text + "_fn" + filename + ".jpg"
            cv.imwrite(jpg_filenameNew, frame)     # save frame as JPEG file

#save to csv file

def save_to_file(resultsArr):
    # change a to w to start new file a to append to end of current file
    with open('spreadsheet_batchf.csv', 'a', newline='') as csvfile:
        fieldnames = ['plate_number', 'confidence', 'video_time', 'file_name', 'creation_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #unquote for headers to be written
        #writer.writeheader()
        writer.writerows(resultsArr)  

                        
#function to analyze video

def analyze_video (whole_path, filename):
        
        resultsArr = []
        checkArr = []
        imgArr= []
        
        # Open the video file (replace with your video file path)
        video_path = whole_path
        cap = cv.VideoCapture(video_path)
        video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        # Frame skipping factor (adjust as needed for performance)
        frame_skip = 3 # change to skip frames
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()  # Read a frame from the video
            
            if not ret:
                break  # Exit loop if there are no frames left

            timeElapsed = round(cap.get(cv.CAP_PROP_POS_MSEC)/1000, 2)

            #print status
            if frame_skip != 0 and frame_count % 30 == 0:
               
                print("analyzing:" + " " + whole_path + " " + str(timeElapsed) + "sec")


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
            
            #print(alpr_results)
            if len(alpr_results) !=0:
                

                # loop through results and add to the dictionary
                
                for x in alpr_results:
                   #if new plate add data to list
                    add_new_plate(x, checkArr, filename, frame, timeElapsed, whole_path, resultsArr, imgArr)

                    # save the image with the highest confidence for each plate
                    
                    best_image(filename, frame, imgArr, x)
                    
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
        save_to_file(resultsArr)
        
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

            #call analysis function for each file
            analyze_video(whole_path, filename)

            
# call get files to start analysis
get_files()
       





        

