import cv2

cap = cv2.VideoCapture("/home/stimmons/Documents/college/fourthYear/fyp/video/raw/DSC_0010.MOV")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../media/video/edited/lengths2.avi',fourcc, 25.0, (720,430), True)

print("Starting write")

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret: break

    frame = cv2.resize(frame, None, fx=0.375, fy=0.4444, interpolation=cv2.INTER_CUBIC)

    frame = frame[50:,:,:]

    print(frame.shape)

    out.write(frame)
    print("Writing")

cap.release()
out.release()
cv2.destroyAllWindows()