import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Variable to control the operation
operation_active = False

def detect_hand_gesture(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Check if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the landmarks for the thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate the distance between thumb and index finger tips
            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            
            # If the distance is small, consider it a fist (closed hand)
            if distance < 0.05:
                return "closed"
            else:
                return "open"
    return "none"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Detect hand gesture
    gesture = detect_hand_gesture(frame)
    
    # Control the operation based on the gesture
    if gesture == "open":
        operation_active = True
        cv2.putText(frame, "Operation: START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif gesture == "closed":
        operation_active = False
        cv2.putText(frame, "Operation: STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


#Explanation:
# MediaPipe Hands: This is used to detect hand landmarks in real-time.

# Gesture Detection: The distance between the thumb tip and index finger tip is calculated. If the distance is small, it is considered a closed hand (fist). Otherwise, it is considered an open hand.

# Operation Control: If the hand is open, the operation starts (you can replace this with any operation you want). If the hand is closed, the operation stops.

# Visual Feedback: The program displays the current state of the operation on the video feed.