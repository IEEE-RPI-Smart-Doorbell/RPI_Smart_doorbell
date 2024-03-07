import cv2

#Load the pre-trained face detection classifiers
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

#Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    #Read the current frame from the webcam
    ret, frame = cap.read()

    #Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect frontal faces in the frame
    frontal_faces = face_cascade_frontal.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    #Detect profile faces in the frame
    profile_faces = face_cascade_profile.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    # Draw rectangles around frontal faces
    for (x, y, w, h) in frontal_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Draw rectangles around profile faces
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with faces detected
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
