import cv2 as cv
import mediapipe as mp
import time

# Capturing vid (change filename to 0 if need webcam)
capture = cv.VideoCapture("videos/vid_test_smile.3gp")

mpFace = mp.solutions.face
face = mpFace.Face()

pTime = 0

while True:
    # Reading currunt frame
    success, img = capture.read()

    # If can't read currunt frame, break loop
    if not success:
          break

    # Convert vid to rgb for mediapipe
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Calculating fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # Show fps
    cv.putText(img, f"Fps: {int(fps)}", (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv.imshow("Video", img)
    key = cv.waitKey(20)

    if key==27:
        break # If key is pressed, break loop

capture.release()
cv.destroyAllWindows()
