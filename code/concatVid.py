import cv2

lengthsVideoPath = "../media/video/edited/lengths.avi"
splashingVideoPath = "../media/video/edited/splashing.avi"

cap1 = cv2.VideoCapture(lengthsVideoPath)
cap2 = cv2.VideoCapture(splashingVideoPath)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../media/video/edited/combo.avi',fourcc, 25.0, (720,430), True)

print("Starting write")

while(cap1.isOpened() or cap2.isOpened()):
    ret1, frame1 = cap1.read()

    if not ret1:
        ret2, frame2 = cap2.read()
        if not ret2: break
        out.write(frame2)
        print("Writing frame 2")
    else:
        out.write(frame1)
        print("Writing frame 1")


cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()