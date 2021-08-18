import face_recognition
import cv2
import numpy as np
import os
import glob
import ntpath
from pathlib import Path
import socket
import time
import matplotlib.pyplot as plt
# Get a reference to webcam #0 (the default one)
try:
    video_capture = cv2.VideoCapture(0)
except:
    print("No Camera Source Found!")

known_face_encodings = []
known_face_names = []
live_face_encodings = []
live_face_names = []
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'known_people/')

# make an array of all the saved jpg images paths
images_paths = [f for f in glob.glob(path+'*.jpg')]

for i in range(len(images_paths)):
     
    # load images to face rec liberary
    # add encodings to known_faces_encodings
    known_face_encodings.append(face_recognition.face_encodings(
        face_recognition.load_image_file(images_paths[i]))[0])
    
    # add names to each face from its path
    images_paths[i] = images_paths[i].replace(".jpg","")
    known_face_names.append(Path(images_paths[i]).name)
    

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    small_frame = small_frame[:, :, ::-1]

    live_face_locations = face_recognition.face_locations(small_frame)
    
   # print(live_face_locations[0])
    
    live_face_encodings = face_recognition.face_encodings(small_frame, live_face_locations)
    live_face_names=known_face_names

    for encodings in live_face_encodings:
         
            # create a measure of similiarty between known faces and live faces
            face_distances = face_recognition.face_distance(known_face_encodings, encodings)
            print(face_distances)
            
            # choose closest
            best_match_index = np.argmin(face_distances)
            
            print(best_match_index)
            
            
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
                live_face_names[best_match_index] = name
            else:
                name= "unkown"
              
            
            print(live_face_names[best_match_index])
                                       
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, ( 25,25), font, 1.0, (255, 0, 0), 1)
            


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
