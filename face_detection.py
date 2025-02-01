import cv2
import os

# Load the Haar cascade file
haar_cascade_path = "haarcascade_frontalface_default.xml"

# Check if the cascade file exists
if not os.path.exists(haar_cascade_path):
    print(f"Error: Haar cascade file not found at '{haar_cascade_path}'.")
    exit()

haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if the cascade file is loaded correctly
if haar_cascade.empty():
    print("Error: Failed to load Haar cascade file.")
    exit()

# Initialize the webcam (use 0 for default camera)
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Start video capture loop
while True:
    # Read a frame from the camera
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture an image from the camera.")
        break

    # Convert the image to grayscale
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = haar_cascade.detectMultiScale(grayimg, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with detected faces
    cv2.imshow("Face Detection", img)

    # Exit the loop when the 'Esc' key is pressed
    key = cv2.waitKey(10)
    if key == 27:  # ASCII code for 'Esc'
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
