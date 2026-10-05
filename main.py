import cv2
import mediapipe as mp

# Initialize the webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

# create face landmarks from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Create FaceMesh ONCE
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def getLandmarks(image):
    # MediaPipe expects RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    
    results = face_mesh.process(image)
    
      # Check that a face was actually detected
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return landmarks, results

    return None, results
            
# Draw the face mesh annotations on the image.
def drawFaceMesh(image, results):
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #         print('face landmarks', face_landmarks)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        
while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab frame. Is your camera turned on?")
        break

    # Display the resulting frame in a window
    cv2.imshow('VS Code Webcam Feed', frame)
    
    # flip the video vertically and change the BGR to RGB
    frame = cv2.flip(frame, 1)
    
    # Detect facial landmarks
    landmarks, results = getLandmarks(frame)

    # Draw face mesh
    drawFaceMesh(frame, results)

    # Display final frame
    cv2.imshow("Drowsiness Detection", frame)
    
    # show webcame frames
    cv2.imshow("Webcam", frame)
    
    # Press 'q' on your keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()