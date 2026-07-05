import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceDetection.process(imgRGB)



    if not success:
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        3
    )

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

