import cv2

max_width = 640
max_height = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,max_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,max_height)

while(cap.isOpened()):
    ret,frame=cap.read() # 사진 찍기 -> (480,640,3)
    if not ret: break

    # 이미지 출력
    cv2.imshow('cam',frame)
    
    # 10ms 동안 키 입력 대기
    key = cv2.waitKey(10)
    if  key == ord('q'): break
    
cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기