# Mini_Project


## 메인 프로그램 (알고리즘 부분)

#### 알고리즘
    - debouncing (prev_state != cur_state)
    - hysterisis
    - 확률로 처리하는게 나을듯
    - 5frame cnt -> state 바꾸고
    -
    - round 5개?
    - 사운드


#### 적용값
    - multiprocessing 으로 frame_queue에 frame을 담아두고, image processing 
    - model load 및 inference GPU로 적용 (interpreter = tflite.Interpreter(model_path=modelPath, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]) # 모델 로딩)
    - threshold = 0.6 (확률 60% 적용)
    - count_win = 10 (10 frame cnt -> state 변경)

    - round 미적용 (while로 처리) -> 추후 변경
    - sound 미적용 (BGM 및 Effect sound 찾아보기) -> 추후 변경

#### 적용한 데이터
 
    - cal_result 함수로 계산 (UI 적용 전까지는 print문으로 출력력)

 timing_list = [100,300, 400, 500, 600 ,700 ,800, 1000]
 age_list = ['신','10대','20대','30대','40대','50대','60대','70대','다시']
 score_list = [100, 90, 80, 70, 60, 50, 40,30,0]