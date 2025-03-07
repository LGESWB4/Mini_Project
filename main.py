# 모듈 로딩
import tflite_runtime.interpreter as tflite
# import tensorflow as tf
import numpy as np
import time
import cv2 
import random as rand
import multiprocessing as mp

# TFLite 모델 로딩
modelPath = 'RPS_PreTrained_SSD.tflite'
#interpreter = tflite.Interpreter(model_path=modelPath, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]) # 모델 로딩
interpreter = tflite.Interpreter(model_path = modelPath) # 모델 로딩
interpreter.allocate_tensors() # tensor 할당rne
input_details = interpreter.get_input_details()  # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
input_dtype = input_details[0]['dtype']
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('model input shape:', (height, width))
#print(input_details)
#print(output_details)

classList = '_ Scissors Rock Paper'.split()
colorList = [(),(255,0,0),(0,255,0),(0,0,255)]


IMG_SIZE = 320
threshold = 0.65 # 모델 정확도

count_win = 10 # 이긴 frame 세팅값 (debouncing)
game_round = 10 # 게임 라운드 세팅값

timing_list = [100, 300, 400, 500, 600 ,700 ,800, 1000]
age_list = ['신','10대','20대','30대','40대','50대','60대','70대','실패']
score_list = [100, 90, 80, 70, 60, 50, 40, 30, 0]

def process_image(frame_queue, action_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        # 1ms 동안 키 입력 대기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        processImage(frame, action_queue)

def processImage(frame, action_queue):
    # frame 크기 저장 -> BB 표시할 때 사용 
    frameH, frameW, _ = frame.shape

    # BGR을 RGB로 변경
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # 모델의 입력 형태로 수정: (1,320,320,3)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    img = np.expand_dims(img, 0)

    # -1 ~ 1 사이 값으로 변경
    img = img * (2/255) - 1

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
    #   모델 실행
    interpreter.invoke()
    #   output tensor 얻기
    bboxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classIndexes = interpreter.get_tensor(output_details[3]['index'])[0].astype(int)
    classScores = interpreter.get_tensor(output_details[0]['index'])[0]

    for bbox, c, cs in zip(bboxes, classIndexes, classScores):
        # score가 threshold 이하이면 skip
        if cs <= threshold: continue

        # score를 100 분율로 환산
        classConfidence = round(cs*100)
        
        # BB 표시
        classIndex = c + 1 # shift 필요: 0 1 2 -> 1 2 3
        action_queue.append(c)

        classLabel = classList[classIndex]
        classColor = colorList[classIndex]
        displayText = f'{classLabel}: {classConfidence}%'.upper()
        ymin, xmin, ymax, xmax = bbox
        xmin, xmax, ymin, ymax = int(xmin*frameW), int(xmax*frameW), int(ymin*frameH), int(ymax*frameH)
        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2)
        cv2.putText(frame, displayText, (xmin, ymin-7), 
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=classColor, thickness=2)         

# reaction_time 에 따른 점수 계산
def cal_result(result_time):
    idx = -1

    if result_time == 0:
        idx = 8
    elif result_time <= timing_list[0]:
        idx = 0
    elif timing_list[0] < result_time <= timing_list[1]:
        idx = 1
    elif timing_list[1] < result_time <= timing_list[2]:
        idx = 2
    elif timing_list[2] < result_time <= timing_list[3]:
        idx = 3
    elif timing_list[3] < result_time <= timing_list[4]:
        idx = 4
    elif timing_list[4] < result_time <= timing_list[5]:
        idx = 5
    elif timing_list[5] < result_time <= timing_list[6]:
        idx = 6
    elif timing_list[6] < result_time <= timing_list[7]:
        idx = 7
    elif result_time > timing_list[7]:
        idx = 8

    age = age_list[idx]
    score = score_list[idx]
    return age, score

def main(frame_queue,action_queue):
    total_score = 0
    total_reaction_time = 0
    cnt = 0
    for cnt in range(1,game_round+1,1):
    #while(True):
        #cnt += 1
        # 랜덤 가위바위보 시작 (컴퓨터)
        computer_action = np.random.randint(0,3)
        # 랜덤 시작 시간 설정
        waiting_time = np.random.randint(2,5)
        time.sleep(waiting_time)

        # real-time frame capture
        result_time = capture_frame(frame_queue,action_queue,computer_action)
        # 사용자가 이기는 time 세기 (s -> ms 변환)
        result_time = result_time*1000/count_win
        # 총 걸린 시간 계산
        total_reaction_time += result_time

        # 결과 계산
        age_val, score_val = cal_result(result_time)
        total_score += score_val
        print("{}라운드 걸린 시간: {:.2f}ms, 획득한 점수: {}점".format(cnt, result_time, score_val))

    # 평균 걸린 시간 계산
    total_reaction_time = total_reaction_time/game_round
    age_val, score_val = cal_result(total_reaction_time)

    print("최종 점수: {}점, 평균 걸린 시간: {:.2f}ms, 당신은? {}".format(total_score, total_reaction_time, age_val))


def set_computer_action_for_user(action_queue, computer_action):
    if action_queue:
        win_action = (computer_action+1)%3    
        if action_queue[-1] == win_action:
            computer_action_org = computer_action

            possible_actions = [0, 1, 2]
            possible_actions.remove(computer_action)
            computer_action = rand.choice(possible_actions)
            print("user_predict_action: {}".format(classList[action_queue[-1]+1]))
            print("computer_action_change {} -> {}".format(classList[computer_action_org+1],classList[computer_action+1]))

    return computer_action

def capture_frame(frame_queue,action_queue, computer_action):
    # 카메라 설정
    cap = cv2.VideoCapture(0) # 0번 카메라 열기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    # 사용자의 현재 상태 읽고, 다른 computer_action 설정 (개발자 의도도)
    ret,frame=cap.read()
    frame_queue.put(frame)
    computer_action = set_computer_action_for_user(action_queue, computer_action)

    # 가위바위보 시작과 동시에 화면에 보여줌, 카메라 띄움
    startTime = 0
    game_start_time = time.time()

    while(cap.isOpened()):
        ret,frame=cap.read() # 과거에 캡처된 이미지 읽어서 버리기
        if not ret: break

        ret,frame=cap.read() # 사진 찍기 -> (320,320,3)
        if not ret: break

        # FPS 계산
        curTime = time.time()
        fps = 1/(curTime - startTime)
        startTime = curTime

        # FPS 표시
        cv2.putText(frame,f'FPS: {fps:.1f}',(20, 50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
        cv2.putText(frame, f'computer: {classList[computer_action+1]}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2,(0,255,255),2)

        # 이미지 출력
        cv2.imshow('cam',frame)
        frame_queue.put(frame)
        
        if action_queue:
            win_action = (computer_action+1)%3

            if action_queue[-1] == win_action:
                result = action_queue.count(win_action)
                if result >= count_win:
                    game_end_time = time.time()
                    break

        # 10초 이상 게임 진행시 라운드 강제 종료
        if time.time() - game_start_time > 10: 
            game_end_time = game_start_time
            break
        # 1ms 동안 키 입력 대기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            game_end_time = game_start_time
            break

    cap.release() # 카메라 닫기
    cv2.destroyAllWindows() # 모든 창 닫기
    while frame_queue.empty() is False:
        frame_queue.get()
    del action_queue[:]

    result_time = game_end_time - game_start_time

    return result_time
if __name__ == "__main__":
    frame_queue = mp.Queue()
    manager = mp.Manager()
    action_queue = manager.list()

    # 게임 진행과 이미지 처리를 병렬로 실행
    p1 = mp.Process(target=main, args=(frame_queue,action_queue))
    p2 = mp.Process(target=process_image, args=(frame_queue,action_queue))

    p1.start()
    p2.start()

    p1.join()
    p2.join()