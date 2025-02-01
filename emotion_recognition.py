import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Analyze the frame and detect emotions
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Get the emotion with the highest score
    emotion = result[0]['dominant_emotion']
    confidence = result[0]['emotion'][emotion]

    # Draw the emotion on the frame
    cv2.putText(frame, f"{emotion}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
