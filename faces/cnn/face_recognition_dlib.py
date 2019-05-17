# import required packages
import cv2
import dlib

face_weights = './mmod_human_face_detector.dat'

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(face_weights)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(frame, 1)

    # loop over detected faces
    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y

    # draw box over face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

