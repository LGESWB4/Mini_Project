import cv2
import utils
import random
from multiprocessing import Process
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class GameScreen(FloatLayout):
    def __init__(self, switch_callback, **kwargs):
        super().__init__(**kwargs)
        self.switch_callback = switch_callback

        # game settings init
        self.round = 0
        self.fps = 0

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

        #self.fps_txt.text = f"FPS: {self.fps}"

        # round text
        self.round_txt = Label(text=f"ROUND: {self.round}",
                 font_size=utils.DESCRIPTION_FONT_SIZE,
                 font_name=utils.FONT_NAME,
                 pos_hint={'center_x': 0.2, 'center_y': 0.9},
                 color=utils.COLOR_WHITE)
        self.add_widget(self.round_txt)

        # status text
        self.status_txt = Label(text="",
                                font_size=utils.SUBTITLE_FONT_SIZE,
                                font_name=utils.FONT_NAME,
                                color=utils.COLOR_RED)
        self.add_widget(self.status_txt)

        # opencv init
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        # game start
        self.start_game()

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame.flatten()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.webcam.texture = texture

    def stop_camera(self):
        self.capture.release()

    def random_rockscissorspaper_img(self):
        rockscissorspaper_img = [utils.ROCK_IMG, utils.SCISSORS_IMG, utils.PAPER_IMG]
        return rockscissorspaper_img[random.randint(0, 2)]

    def start_game(self):
        self.round = 0
        self.next_round(0)

    def next_round(self, dt):
        if hasattr(self, 'rsp_img'):
            self.remove_widget(self.rsp_img)

        if self.round < 5:
            self.round += 1
            self.round_txt.text = f"ROUND: {self.round}"
            self.status_txt.text = "Ready"
            Clock.schedule_once(self.show_start, 5)
        else:
            self.switch_callback(None)

    def show_start(self, dt):
        self.status_txt.text = "Start"
        wait_time = random.uniform(1, 3)
        Clock.schedule_once(self.show_rsp, wait_time)

    def show_rsp(self, dt):
        self.status_txt.text = ""
        self.rsp_img = Image(size_hint=(0.3, 0.3), pos_hint={'center_x': 0.2, 'center_y': 0.5})
        self.rsp_img.source = self.random_rockscissorspaper_img()
        self.add_widget(self.rsp_img)
        Clock.schedule_once(self.next_round, 5)
