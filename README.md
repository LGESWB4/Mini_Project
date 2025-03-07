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
    - multiprocessing 으로 frame_queue에 frame을 담아두고, image processing 으로 action_queue 에 inference 값 담아둠. (이를 process 간 공유 변수 (Queue, List)를 지정하여 공유 )
    - model load 및 inference GPU로 적용 해야함 (현재 CPU Multiprocessing)
    - threshold = 0.65 (확률 65% 적용)
    - count_win = 10 (10 frame cnt -> state 변경)

    - round 총 10라운드 적용 (5라운드는 조금 적은 듯)
    - sound 미적용 (BGM 및 Effect sound 찾아보기) -> 추후 변경

    - 개발자의 의도 (미리 예측하여 사용자가 내고 있다면, (근데 그 값이 컴퓨터를 이길 값이라면) 컴퓨터 값 변경)


#### 적용한 데이터
 
    - cal_result 함수로 계산 (UI 적용 전까지는 print문으로 출력력)

 timing_list = [100,300, 400, 500, 600 ,700 ,800, 1000]
 age_list = ['신','10대','20대','30대','40대','50대','60대','70대','다시']
 score_list = [100, 90, 80, 70, 60, 50, 40,30,0]