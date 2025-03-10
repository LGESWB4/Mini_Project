import cv2
import utils
import numpy as np

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