import cv2 as cv
import mediapipe as mp

# Capturing vid (cange filename to 0 if need webcam)
capture = cv.VideoCapture("videos/vid_test_smile.3gp")

while True:
    # Reading currunt frame
    success, frame = capture.read()

    # If can't read currunt frame, break loop
    if not success:
          break

    cv.imshow("Video", frame)
    key = cv.waitKey(1)

    if key==27:
        break # If key is pressed, break loop

capture.release()
cv.destroyAllWindows()
