"""
Cognition

Developed by Carson Woods
"""
# Import
import numpy as np
import cv2
import argparse
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image



# Set arguments for improving command line functionality
parser = argparse.ArgumentParser(description='Use arguments to change behavior of Cognition')
parser.add_argument('--collect',
                    action="store_true",
                    help="Cognition will store face data")
args = parser.parse_args()

cascadePath = "opencv_cascades/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)


# Create VideoCapture CV2 object on default capture device.
video_capture = cv2.VideoCapture(0)
if video_capture.isOpened():
    print("Camera Intialized Successfully")

# Count to name files stored
count = 0

# Frame Recording Loop. Records frames of images and uses the cascade to classify faces.
while True:
    # Using read rather than get so it automatically decodes. Uses ffmpeg for image decoding.
    # Additionally retVal is important if reading from file rather than camera.
    # FPS is captured at max FPS of camera, however face recognition is limited by CPU speed.
    retVal, frame = video_capture.read()

    # Converts each captured frame to greyScale as a form of preprocessing
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """
        detectMultiScale is a function for detecting objects and is called on the cascade
        that it uses to ID what it is detecting (in this case faces.)
        We pass it the greyscale frame, scaleFactor, minNeighbors, minSize, flags, respectively.
        More information about those parameters can be found online. These values can be modified without
        risk of causing compile errors, but algorithm may become more/less accurate.

        detectMultiScale returns an array of rectangles that it believes contains faces.
    """
    faceArray = faceCascade.detectMultiScale(
        grayFrame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Prints current amount of faces found in current frame
    # print("Found {0} faces!".format(len(faceArray)), end='\r')


    # Draws rectangle on frame so user can see live results
    for (x, y, w, h) in faceArray:
        ROI = frame[y:y+h, x:x+w]
        # If collect argument is passed then data is written to training_data/
        if args.collect:
            # If we extract Region of Interest [ROI] from image then it is preprocessed
            # For future training and we need to do minimal cleaning
            count += 1
            name = "data/Carson_Woods/%d.jpg"%count
            cv2.imwrite(name, ROI)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)


        # Load Keras Model
        classifier = load_model('./models/cognition_model.h5')
        classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Set ROI to image object. Scale to 128x128.
        face = Image.fromarray(ROI, 'RGB')
        size = 150,150
        face.thumbnail(size, Image.ANTIALIAS)

        # Convert Image object back to numpy array.
        test_face = image.img_to_array(face)
        test_face = np.expand_dims(test_face, axis = 0)


        #predict the result
        result = classifier.predict(test_face)

        print(result[0])

        if result[0] >= .5 and len(faceArray) != 0:
            cv2.putText(frame, "Detected: Carson Woods",
                        (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,0,0), 2)
        else:
            cv2.putText(frame, "No Known Face Detected",
                        (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,0,0), 2)

    # Show the frame that has been drawn on
    cv2.imshow("Cognition", frame)

    # If user types q the program exits
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User Shutdown Signal Recieved. Shutting Down...")
        break;

# Destroy video_capture object. Removes reference to capture device.
video_capture.release()
if video_capture.isOpened() == False:
    print("Camera Deinitialized Successfully")
