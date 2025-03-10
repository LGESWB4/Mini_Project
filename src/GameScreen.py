import cv2
import utils
import random as rand
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import time as time
import GameSetting as GS
import ProcessingImage as PI
import numpy as np
from functools import partial
import multiprocessing as mp

class GameScreen(FloatLayout):
    def __init__(self, switch_callback, **kwargs):
        super().__init__(**kwargs)
        self.frame_queue = mp.Queue()
        self.manager = mp.Manager()
        self.action_queue = self.manager.list()

        self.switch_callback = switch_callback

        # game settings init
        self.round = 0
        self.total_round = utils.GAME_ROUND # 총 게임 라운드 수
        self.fps = 0
        self.startTime = 0

        self.total_score = 0
        self.total_reaction_time = 0
        self.game_end_time = 0
        self.game_start_time = 0
        self.result_time = 0
        self.computer_action = 0

        self.frame = 0
        self.max_calls = 300
        self.calls = 0

        # webcam screen init
        self.webcam = Image(size_hint=(0.8, 1), pos_hint={'center_x': 0.7, 'center_y': 0.5})
        self.add_widget(self.webcam)

        # fps text
        self.fps_txt = Label(text=f"FPS: {self.fps}",
                 font_size=utils.DESCRIPTION_FONT_SIZE,
                 font_name=utils.FONT_NAME,
                 pos_hint={'center_x': 0.1, 'center_y': 0.9},
                 color=utils.COLOR_WHITE)
        self.add_widget(self.fps_txt)

        # round text
        self.round_txt = Label(text=f"ROUND: {self.round}",
                 font_size=utils.DESCRIPTION_FONT_SIZE,
                 font_name=utils.FONT_NAME,
                 pos_hint={'center_x': 0.2, 'center_y': 0.9},
                 color=utils.COLOR_WHITE)
        self.add_widget(self.round_txt)

        # score text
        self.score_txt = Label(text=f"SCORE: {self.total_score}",
                 font_size=utils.DESCRIPTION_FONT_SIZE,
                 font_name=utils.FONT_NAME,
                 pos_hint={'center_x': 0.3, 'center_y': 0.9},
                 color=utils.COLOR_WHITE)
        self.add_widget(self.score_txt)

        # status text
        self.status_txt = Label(text="",
                                font_size=utils.SUBTITLE_FONT_SIZE,
                                font_name=utils.FONT_NAME,
                                color=utils.COLOR_RED)
        self.add_widget(self.status_txt)

        # round result text
        self.round_result_txt = Label(text="",
                                font_size=utils.SUBTITLE_FONT_SIZE,
                                font_name=utils.FONT_NAME,
                                pos_hint={'center_x': 0.3, 'center_y': 0.9},
                                color=utils.COLOR_WHITE)
        self.add_widget(self.round_result_txt)

        # openvc camera init
        self.capture = cv2.VideoCapture(0) # 0번 카메라 열기
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera_process = Clock.schedule_interval(self.camera_Update, 1.0/30.0) # 30fps

        # game start
        self.start_game()

    def init_round(self):
        self.game_end_time = 0
        self.game_start_time = 0
        self.result_time = 0
        self.calls = 0
        

        while self.frame_queue.empty() is False:
            self.frame_queue.get()
        del self.action_queue[:]

    def stop_camera(self):
        self.capture.release() # 카메라 닫기
        
    def random_rockscissorspaper_img(self):
        rockscissorspaper_img = [utils.SCISSORS_IMG, utils.ROCK_IMG, utils.PAPER_IMG]
        return rockscissorspaper_img[self.computer_action]

    def start_game(self):
        self.round = 0
        self.next_round(0)

    def next_round(self, dt):
        if hasattr(self, 'rsp_img'):
            self.remove_widget(self.rsp_img)

        self.init_round()   # 데이터 초기화 (라운드 초기화)

        if self.round < self.total_round:
            self.round += 1
            self.round_txt.text = f"ROUND: {self.round}"
            self.status_txt.text = "Ready"
            Clock.schedule_once(self.show_start, self.total_round)
        else:
            self.stop_camera()   # 카메라 종료
            self.camera_process.cancel() # 카메라 프로세스 종료
            self.switch_callback(self.total_score, self.total_reaction_time)

    def show_start(self, dt):
        self.status_txt.text = "Start"

        # 랜덤 시작 시간 설정
        wait_time = rand.uniform(1, 3)
        Clock.schedule_once(self.show_rsp, wait_time)

    def show_rsp(self, dt):
        self.status_txt.text = ""
        self.rsp_img = Image(size_hint=(0.3, 0.3), pos_hint={'center_x': 0.2, 'center_y': 0.5})
        # 랜덤 값 생성 (컴퓨터)
        self.computer_action = np.random.randint(0,3)

        # real-time frame capture
        self.capture_frame(self.after_capture_frame)

    def after_capture_frame(self):
        # 사용자가 이기는 time 세기 (s -> ms 변환)
        self.result_time = self.result_time*1000/utils.COUNT_WIN
        # 총 걸린 시간 계산
        self.total_reaction_time += self.result_time

        # 결과 계산
        age_val, score_val = GS.cal_result(self.result_time)
        self.total_score += score_val
        self.score_txt.text = f"SCORE: {self.total_score}"
        
        print("{}라운드 걸린 시간: {:.2f}ms, 획득한 점수: {}점".format(self.round, self.result_time, score_val))

        # 해당 라운드의 결과 출력 (맨 아래화면 부분에)
        
        Clock.schedule_once(self.next_round, self.total_round)

    def set_computer_action_for_user(self):
        ret,frame=self.capture.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_queue.put(frame)
        self.action_queue = PI.processImage(frame, self.action_queue, 0)

        if self.action_queue:
            win_action = (self.computer_action+1)%3    
            if self.action_queue[-1] == win_action:
                computer_action_org = self.computer_action

                possible_actions = [0, 1, 2]
                possible_actions.remove(self.computer_action)
                self.computer_action = rand.choice(possible_actions)
                print("user_predict_action: {}".format(utils.CLASS_LIST[self.action_queue[-1]+1]))
                print("computer_action_change {} -> {}".format(utils.CLASS_LIST[computer_action_org+1],utils.CLASS_LIST[self.computer_action+1]))
    
    def camera_Update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame.flatten()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.webcam.texture = texture

    def update(self, dt):
        self.calls += 1
        #print(self.calls)
        if self.calls >= self.max_calls:  # Stop after max_calls
            self.game_end_time = self.game_start_time
            self.after_update()
            return
            
        ret, frame = self.capture.read()
        if ret:
            # FPS 계산
            curTime = time.time()
            self.fps = 1/(curTime - self.startTime)
            self.startTime = curTime

            self.fps_txt.text = f"FPS: {self.fps:.2f}"

            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame.flatten()
            """
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.webcam.texture = texture"
            """
            self.frame_queue.put(frame)

            #self.action_queue = PI.processImage(frame, self.action_queue)
            Clock.schedule_once(partial(PI.processImage, frame, self.action_queue), 0)

            if self.action_queue:
                win_action = (self.computer_action+1)%3

                if self.action_queue[-1] == win_action:
                    result = self.action_queue.count(win_action)
                    print("ROUND WIN COUNT: {}".format(result))
                    if result >= utils.COUNT_WIN:
                        self.game_end_time = time.time()
                        #self.update_process.cancel()
                        self.after_update()
                        return

        # 1ms 동안 키 입력 대기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.game_end_time = self.game_start_time
            #self.update_process.cancel()
            self.after_update()
            return

    def capture_frame(self, callback):
        # 사용자의 현재 상태 읽고, 다른 computer_action 설정 (개발자 의도도)
        self.set_computer_action_for_user()

        # 가위바위보 시작과 동시에 화면에 보여줌, 카메라 띄움
        self.startTime = 0
        self.rsp_img.source = self.random_rockscissorspaper_img()
        self.add_widget(self.rsp_img)
        self.game_start_time = time.time()

        self.update_process = Clock.schedule_interval(self.update, 1.0/30.0)  # 30fps

        # Call the callback function after capture_frame is done
        self.callback = callback

    def after_update(self):
        self.update_process.cancel()
        self.fps_txt.text = f"FPS: 0"
        if self.callback:
            self.result_time = self.game_end_time - self.game_start_time
            self.callback()
            self.callback = None