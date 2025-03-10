import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import multiprocessing as mp
import warnings

warnings.simplefilter("ignore", UserWarning)

# 모델 경로 설정
model_path = "/home/willtek/Documents/mini_project/weights/MobileNetV3A_RP_Hard_LossAll.tflite"

# 클래스 매핑
ansToText = {0: 'rock', 1: 'scissors', 2: 'paper'}
colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 카메라 설정
WIDTH, HEIGHT = 400, 300  # 원하는 해상도로 설정

# ✅ 프레임 캡처 프로세스
def capture_frames(frame_queue):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPEG 포맷 사용
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 30FPS 설정

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # 좌우 반전

        # 최신 프레임 유지 (큐가 가득 차면 이전 프레임 삭제)
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

    cap.release()

# ✅ 이미지 전처리 함수
def preprocess_frame(frame):
    """카메라 프레임을 모델 입력 크기에 맞게 변환"""
    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)  # 빠른 리사이징
    frame = frame[..., ::-1]  # BGR → RGB 변환 (cvtColor 대신 NumPy 사용)
    frame = (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
    return np.expand_dims(frame, axis=0)  # (1, C, H, W) 형태로 변환

# ✅ Softmax 함수
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 오버플로 방지
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ✅ 추론 프로세스
def inference(frame_queue, result_queue):
    try:
        # GPU Delegate 설정
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate('libtensorflowlite_delegate_gpu.so')]
        )
        print("✅ GPU 가속 활성화됨")
    except:
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
        print("🖥️ CPU로 실행")

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']

    start_time = time.time()
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # FPS 계산
            cur_time = time.time()
            fps = 1 / (cur_time - start_time)
            # print(f"Latency: {(cur_time - start_time):.2f}")
            print(f"Queue size: {frame_queue.qsize()}")
            start_time = cur_time

            # 이미지 전처리
            input_data = preprocess_frame(frame)

            # 모델 입력 설정 및 실행
            interpreter.set_tensor(input_details[0]['index'], input_data.astype(input_dtype))
            interpreter.invoke()

            # 결과 가져오기
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            probabilities = softmax(output_data)  # Softmax 적용
            ans = np.argmax(probabilities)
            confidence = probabilities[ans]

            # 50% 미만이면 invalid 처리
            if confidence < 0.5:
                text, color = "invalid", (0, 0, 0)  # 검은색
            else:
                text, color = ansToText[ans], colorList[ans]

            # 결과 전송
            result_queue.put((frame, text, color, fps))

# ✅ 메인 실행 프로세스
def main():
    mp.set_start_method("spawn", force=True)  # 멀티프로세싱 초기화

    # 공유 큐 생성
    frame_queue = mp.Queue(maxsize=5)  # 최신 프레임만 유지
    result_queue = mp.Queue()

    # 프로세스 실행
    capture_process = mp.Process(target=capture_frames, args=(frame_queue,))
    inference_process = mp.Process(target=inference, args=(frame_queue, result_queue))

    capture_process.start()
    inference_process.start()

    while True:
        if not result_queue.empty():
            frame, text, color, fps = result_queue.get()

            # 예측 결과 출력
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f'FPS: {fps:.1f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 화면 출력
            cv2.imshow("Webcam", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 프로세스 종료
    capture_process.terminate()
    inference_process.terminate()
    capture_process.join()
    inference_process.join()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
