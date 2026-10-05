import cv2
import mediapipe as mp

# Initialize the webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

# Create face landmarks from mediapipe
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

# ISOLATE LEFT AND RIGHT EYE
# Right eye region
def getRightEye(image, landmarks):
    eye_top = int(landmarks[263].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])
    right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye

# Get the right eye coordinates on the actual -> to visualize the bbox
def getRightEyeRect(image, landmarks):
    eye_top = int(landmarks[257].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])

    cloned_image = image.copy()
    cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_right_eye.shape
    x = eye_left
    y = eye_top
    return x, y, w, h

# Left eye region
def getLeftEye(image, landmarks):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])
    left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye

# Get the left eye coordinates on the actual -> to visualize the bbox
def getLeftEyeRect(image, landmarks):
    # eye_left landmarks (27, 23, 130, 133) ->? how to utilize z info
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])

    cloned_image = image.copy()
    cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_left_eye.shape

    x = eye_left
    y = eye_top
    return x, y, w, h

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
    
    # Only work with eyes if a face was detected
    if landmarks is not None:

        # Right eye
        rightEyeImg = getRightEye(frame, landmarks)
        rightEyeHeight, rightEyeWidth, _ = rightEyeImg.shape
        xRightEye, yRightEye, rightEyeWidth, rightEyeHeight = getRightEyeRect(
            frame, landmarks
        )
        cv2.rectangle(
            frame,
            (xRightEye, yRightEye),
            (xRightEye + rightEyeWidth, yRightEye + rightEyeHeight),
            (200, 21, 36),
            2
        )

        # Left eye
        leftEyeImg = getLeftEye(frame, landmarks)
        leftEyeHeight, leftEyeWidth, _ = leftEyeImg.shape
        xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight = getLeftEyeRect(
            frame, landmarks
        )

        cv2.rectangle(
            frame,
            (xLeftEye, yLeftEye),
            (xLeftEye + leftEyeWidth, yLeftEye + leftEyeHeight),
            (200, 21, 36),
            2
        )

    
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