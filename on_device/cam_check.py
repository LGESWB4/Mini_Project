import cv2

# cap = cv2.VideoCapture(0)  # 웹캠 사용

# # 웹캠이 지원하는 FPS 확인
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"📷 웹캠에서 제공하는 FPS: {fps}")

# cap.release()


###
import cv2
import time

cap = cv2.VideoCapture(0)

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time

    # 1초마다 FPS 측정
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        print(f"📷 실제 FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
