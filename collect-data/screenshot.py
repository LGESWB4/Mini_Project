import cv2, time
import sys

selected = sys.argv[1]
files_dir = './datas/'+selected+'/'
SELECTED_CAMERA = 0
MAX_WIDTH = 640
MAX_HEIGHT = 480
INTERVAL = 2
START_INTERVAL = INTERVAL*5

cap = cv2.VideoCapture(SELECTED_CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,MAX_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,MAX_HEIGHT)

startTime = time.time()*INTERVAL
flag=0
count=0
start=False
while(cap.isOpened()):
    ret,frame=cap.read() # 사진 찍기 -> (480,640,3)
    if not ret: break

    #10초 동안 100장 스크린샷
    curTime = time.time()*INTERVAL
    interval=int(curTime-startTime)
    if interval>START_INTERVAL:
        start=True
    if start and interval>flag:
        flag=interval
        count+=1
        cv2.imwrite(files_dir+selected+str(time.time())+'.png', frame)
    if start:
        cv2.putText(frame,f'Count: {count}',(50, 50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
    else:
        cv2.putText(frame,f'Start in {5-interval//2}',(50, 50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

    # 이미지 출력
    cv2.imshow('cam',frame)
    
    # 10ms 동안 키 입력 대기
    key = cv2.waitKey(10)
    if  key == ord('q') or count==100: break
    
cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기