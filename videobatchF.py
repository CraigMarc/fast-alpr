#video batchf same as video batch expect the code is split up into more functions can eventually delete video batch

import cv2 as cv
from fast_alpr import ALPR
import sys
import csv
import os
import time
import datetime
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

### check if plate has been found in the last 3 minutes so dont save duplicates
def check_files (directory, current_file, plate_number):
    
    for file in os.listdir(directory):
       
        if plate_number in file:
            
            compare_current = (current_file[0:8])
         
            end_file = file[-16:]
            compare_matched = end_file[0:8]
           
            difference = int(compare_current) - int(compare_matched)
            
            if difference == 0:
                return "false"
            
            if difference < 30000 and difference != 0:
                return "true"
    
    return "false"
            

# get creation time of video file

def get_creation_time (whole_path):
    

        ti_m = os.path.getmtime(whole_path)
        
        # Converting the time in seconds to a timestamp
        creation_time = time.ctime(ti_m)
        return creation_time

# get date and month of file to name folder

def get_folder_time (whole_path):

    ti_m = os.path.getmtime(whole_path)
    folder_time = datetime.datetime.fromtimestamp(ti_m)
    folder_time_format = folder_time.strftime("%Y%m%d")
    
    return folder_time_format
       
# function to add new plate

def add_new_plate (x, checkArr, filename, frame, timeElapsed, whole_path, resultsArr, imgArr, directory_2):

    if x.ocr.text not in checkArr and x.ocr.confidence >= 0.95:

        creation_time = get_creation_time(whole_path)
        folder_time = get_folder_time(whole_path)

        # other files may save later ???????????????????   
        #jpg_filename = "jpeg/" + filename[:-4] + str(frame_count) + ".jpg"
        #jpg_filename = "C:/Users/Criag/Videos/jpeg_files/" + x.ocr.text + "_c" + str(int(x.ocr.confidence * 100000)) + "_fn" + filename + ".jpg"
        #cv.imwrite(jpg_filename, frame)     # save frame as JPEG file 
        

        # save first plate image to jpeg_best folder
        
        #create folder if it does not exist
        print("detected:  " + "plate_number: " + x.ocr.text + " confidence: " + str(x.ocr.confidence))
        if not os.path.exists("C:/fast_alpr/jpeg_best/" + folder_time + "/"):
            os.makedirs("C:/fast_alpr/jpeg_best/" + folder_time + "/")

        file_check = check_files("C:/fast_alpr/jpeg_best/" + folder_time + "/", filename, x.ocr.text)
       
        #save first occurance of plate to file
        if file_check == "false":
            jpg_filenameFirst = "C:/fast_alpr/jpeg_best/" + folder_time + "/" + x.ocr.text + "_fn" + filename + ".jpg"
            cv.imwrite(jpg_filenameFirst, frame)     # save frame as JPEG file

            #add plate data to results array to save at end of video
       
            data = {
            "plate_number": x.ocr.text,
            "confidence": round(x.ocr.confidence, 3),
            "video_time": timeElapsed,
            "file_name": whole_path,
            "creation_time": creation_time
            }

            resultsArr.append(data)

        # add plate data an confidence to array so can be used to get best image 
        chkData = {
        "plate_number": x.ocr.text,
        "confidence": x.ocr.confidence
        }
        
        imgArr.append(chkData)

        # add plate number to array so dont have duplicate plate numbers    
        checkArr.append(x.ocr.text)

def best_image (filename, frame, imgArr, x, whole_path):

   #see if best plate number if it is save the nem image
    for iter in imgArr:
        if iter["plate_number"] == x.ocr.text and iter["confidence"] < x.ocr.confidence:
            print("detected:  " + "plate_number: " + x.ocr.text + " confidence: " + str(x.ocr.confidence))
            folder_time = get_folder_time(whole_path)
            file_check = check_files("C:/fast_alpr/jpeg_best/" + folder_time + "/", filename, x.ocr.text)
            iter["confidence"] = x.ocr.confidence

            if file_check == "false":
                jpg_filenameNew = "C:/fast_alpr/jpeg_best/" + folder_time + "/" + x.ocr.text + "_fn" + filename + ".jpg"
                cv.imwrite(jpg_filenameNew, frame)     # save frame as JPEG file

#save to csv file

def save_to_file(resultsArr):
    # if file does not exist create file with headers
    if os.path.isfile("C:/fast_alpr/spreadsheet_batchf.csv") == False:
        
        with open('C:/fast_alpr/spreadsheet_batchf.csv', 'w', newline='') as csvfile:
            fieldnames = ['plate_number', 'confidence', 'video_time', 'file_name', 'creation_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(resultsArr) 
    # if file exists append to current file
    else:
        
        with open('C:/fast_alpr/spreadsheet_batchf.csv', 'a', newline='') as csvfile:
            fieldnames = ['plate_number', 'confidence', 'video_time', 'file_name', 'creation_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(resultsArr)  

                        
#function to analyze video

def analyze_video (whole_path, filename, directory_2):
        
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
                    add_new_plate(x, checkArr, filename, frame, timeElapsed, whole_path, resultsArr, imgArr, directory_2)

                    # save the image with the highest confidence for each plate
                    
                    best_image(filename, frame, imgArr, x, whole_path)
                    
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
        
        # save results to csv file at the end of video
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
        
        #loop through files in directory
        for file in os.listdir(directory_2):

            filename = os.fsdecode(file)
            whole_path = directory_2 + "\\" + filename
            
            #call analysis function for each file
            analyze_video(whole_path, filename, directory_2)
            
            
# call get files to start analysis
get_files()
       





        

