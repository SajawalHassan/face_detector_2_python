import cv2 as cv
import mediapipe as mp
import time

# Capturing vid (change filename to 0 if need webcam)
capture = cv.VideoCapture("videos/vid_test_smile.3gp")

mpFaceDetection = mp.solutions.face_detection
FaceDetection = mpFaceDetection.FaceDetection(0.75)

pTime = 0

while True:
    # Reading currunt frame
    success, img = capture.read()

    # If can't read currunt frame, break loop
    if not success:
          break

    # Convert vid to rgb for mediapipe
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Feed RGB img to mediapipe
    results = FaceDetection.process(imgRGB)

    # If you find a face
    if results.detections:
        for id, face in enumerate(results.detections):
            ih, iw, ic = img.shape # Get dimensions of video
            bboxC = face.location_data.relative_bounding_box
            
            # Convert coordinates to pixels (needed for opencv)
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            cv.rectangle(img, bbox, (0,0,0), 2) # Draw rectangle around detected face

            cv.putText(img, f"Accuraccy: {int(face.score[0]*100)}%", (bbox[0], bbox[1]-5),
             cv.FONT_HERSHEY_COMPLEX, .5, (0,255,0), 1) # Showing accuraccy

    # Calculating fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # Show fps
    cv.putText(img, f"Fps: {int(fps)}", (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv.imshow("Video", img)
    key = cv.waitKey(0)

    if key==27:
        break # If key is pressed, break loop

capture.release()
cv.destroyAllWindows()
