import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import os

# source /home/willtek/work/env/bin/activate

# ignore subnormal warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

# use_small_model = True
# model_path = "/home/willtek/Documents/mini_project/weights/sample.tflite"

use_small_model = False  # True: (1, 64, 64, 3), False: (1, 3, 300, 400)
model_path = "/home/willtek/Documents/mini_project/weights/best_epoch.tflite"
# model_path = "/home/willtek/Documents/mini_project/weights/final.tflite"

# GPU Delegate 설정 (라즈베리파이에서 GPU 사용)
try:
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate('libtensorflowlite_delegate_gpu.so')]
    )
    print("✅ GPU 가속 활성화됨")
except:
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    # interpreter = tflite.Interpreter(model_path=model_path)
    print("🖥️ CPU로 실행")

interpreter.allocate_tensors()

# 모델 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
input_shape = input_details[0]['shape']

# 모델 입력 크기 설정
if use_small_model:
    batch_size, height, width, channels = input_shape  # (1, 64, 64, 3)
else:
    batch_size, channels, height, width = input_shape  # (1, 3, 300, 400)

print(f"📏 Model Input Shape: {input_shape}")

# 클래스 매핑
ansToText = {0: 'rock', 1: 'scissors', 2: 'paper'}
colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 저장할 디렉토리 생성
output_dir = "frames_output"
os.makedirs(output_dir, exist_ok=True)

# 카메라 설정
cap = cv2.VideoCapture(0)  # 웹캠 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 저장할 프레임 개수
frame_count = 5
saved_frames = 0

def preprocess_frame(frame):
    """카메라 프레임을 모델 입력 크기에 맞게 변환"""
    frame = cv2.resize(frame, (width, height))  # 모델 입력 크기로 조정
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
    frame = frame.astype(np.float32) / 255.0  # 정규화
    
    if use_small_model:
        frame = np.expand_dims(frame, axis=0)  # (1, 64, 64, 3)
    else:
        frame = np.transpose(frame, (2, 0, 1))  # (H, W, C) → (C, H, W)
        frame = np.expand_dims(frame, axis=0)  # (1, C, H, W)
    
    return frame

def softmax(logits):
    """Softmax 함수 적용"""
    exp_logits = np.exp(logits - np.max(logits))  # 오버플로우 방지
    return exp_logits / np.sum(exp_logits)

while cap.isOpened() and saved_frames < frame_count:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전 (웹캠 자연스러운 방향)

    # 전체 프레임 처리 시작 시간
    start_time = time.time()

    # 이미지 전처리
    input_data = preprocess_frame(frame)

    # 모델 입력 설정
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(input_dtype))

    # 모델 추론 시작 시간
    inference_start = time.time()
    interpreter.invoke()
    inference_end = time.time()

    # 추론 Latency 계산 (ms)
    inference_time = (inference_end - inference_start) * 1000  # ms 단위

    # 결과 가져오기 (logits)
    logits = interpreter.get_tensor(output_details[0]['index'])[0]

    # Softmax 적용
    softmax_probs = softmax(logits)

    # 가장 높은 확률의 클래스 선택
    ans = np.argmax(softmax_probs)
    confidence = softmax_probs[ans]  # 확률값

    # 50% 미만 확률이면 Invalid 처리
    if confidence < 0.5:
        text = "Invalid"
        text_color = (255, 255, 255)  # 흰색
    else:
        text = f"{ansToText[ans]} ({confidence * 100:.1f}%)"
        text_color = colorList[ans]

    # 전체 프레임 처리 종료 시간
    end_time = time.time()
    total_time = end_time - start_time

    # FPS 계산
    fps = 1 / total_time

    # 예측 결과 표시
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {text} | FPS: {fps:.1f} | Latency: {inference_time:.2f}ms")
    plt.axis("off")

    # 프레임 저장
    save_path = os.path.join(output_dir, f"frame_{saved_frames + 1}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {save_path} | FPS: {fps:.1f} | Latency: {inference_time:.2f}ms | Confidence: {confidence:.2f}")
    print(f"Softmax: {softmax_probs}")  # Softmax 확률 출력

    saved_frames += 1

cap.release()
print("🎉 모든 프레임 저장 완료!")
