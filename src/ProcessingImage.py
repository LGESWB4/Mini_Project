import cv2
import utils
import numpy as np
import time 
def processImage(frame, action_queue, dt):
    # frame 크기 저장 -> BB 표시할 때 사용 
    frameH, frameW, _ = frame.shape

    # BGR을 RGB로 변경
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # 모델의 입력 형태로 수정: (1,320,320,3)
    img = cv2.resize(img, (utils.IMG_SIZE,utils.IMG_SIZE))
    img = np.expand_dims(img, 0)

    # -1 ~ 1 사이 값으로 변경
    img = img * (2/255) - 1

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    utils.INTERPRETER.set_tensor(utils.INPUT_DETAILS[0]['index'], img.astype(utils.INPUT_DTYPE))
    #   모델 실행
    utils.INTERPRETER.invoke()
    #   output tensor 얻기
    bboxes = utils.INTERPRETER.get_tensor(utils.OUTPUT_DETAILS[1]['index'])[0]
    classIndexes = utils.INTERPRETER.get_tensor(utils.OUTPUT_DETAILS[3]['index'])[0].astype(int)
    classScores = utils.INTERPRETER.get_tensor(utils.OUTPUT_DETAILS[0]['index'])[0]

    for bbox, c, cs in zip(bboxes, classIndexes, classScores):
        # score가 threshold 이하이면 skip
        if cs <= utils.THRESHOLD: continue

        # score를 100 분율로 환산
        classConfidence = round(cs*100)
        
        # BB 표시
        classIndex = c + 1 # shift 필요: 0 1 2 -> 1 2 3
        action_queue.append(c)

    return action_queue

def preprocess_frame(frame):
    """카메라 프레임을 모델 입력 크기에 맞게 변환"""
    frame = cv2.resize(frame, (utils.WIDTH, utils.HEIGHT), interpolation=cv2.INTER_AREA)  # 빠른 리사이징
    frame = frame[..., ::-1]  # BGR → RGB 변환 (cvtColor 대신 NumPy 사용)
    frame = (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
    return np.expand_dims(frame, axis=0)  # (1, C, H, W) 형태로 변환

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 오버플로 방지
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def inference(frame, action_queue, dt):
    # 이미지 전처리
    input_data = preprocess_frame(frame)
    #s_time = time.time()
    # 모델 입력 설정 및 실행
    #print("inference before time{}".format(s_time))
    utils.INTERPRETER.set_tensor(utils.INPUT_DETAILS[0]['index'], input_data.astype(utils.INPUT_DTYPE))
    utils.INTERPRETER.invoke()

    # 결과 가져오기
    output_data = utils.INTERPRETER.get_tensor(utils.OUTPUT_DETAILS[0]['index'])[0]

    #e_time = time.time()
    #print("inference after time{}".format(e_time-s_time))

    probabilities = softmax(output_data)  # Softmax 적용
    ans = np.argmax(probabilities)
    confidence = probabilities[ans]

    # 50% 미만이면 invalid 처리
    if confidence < utils.THRESHOLD:
        text, color = "invalid", (0, 0, 0)  # 검은색
        ans = -1
    else:
        text, color = utils.ANSTOTEXT[ans], utils.COLORLIST[ans]

        #print("inference: {}, conf: {}".format(utils.MODEL_TO_ANS[ans], confidence))

        # 결과 전송
    action_queue.append(utils.MODEL_TO_ANS[ans])