import numpy as np
import cv2
import os
from PIL import Image

dataName = "Carson_Woods"

cascadePath = "opencv_cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
size = 150,150

try:
    os.mkdir("./data/"+dataName+"_processed/")
except:
    print("Directory already exists")

i = 0

for filename in os.listdir("data/"+dataName+"/"):
    # Load an color image in grayscale
    frame = cv2.imread(os.getcwd()+"/data/"+dataName+"/"+filename,0)
    i+=1
    print(i)

    faceArray = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faceArray:
        ROI = frame[y:y+h, x:x+w]
        name = "data/"+dataName+"_processed/"+filename
        cv2.imwrite(name, ROI)

for filename in os.listdir("./data/"+dataName+"_processed/"):
    outfile = os.getcwd() + "/data/"+dataName+"_processed/" + filename
    try:
        im = Image.open("./data/"+dataName+"_processed/" + filename)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(outfile, "JPEG")
    except IOError:
        if filename != ".DS_Store":
            print("Cannot resize '%s'" % filename)
