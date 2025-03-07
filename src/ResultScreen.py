from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
import utils
import random

class ResultScreen(FloatLayout):
    def __init__(self, switch_callback_main, switch_callback_game, **kwargs):
        super().__init__(**kwargs)

        self.age = '10대'
        self.response_time = 100
        self.response_title = utils.RESPONSE_TIME_TITLE[utils.AGE_LIST.index(self.age)]
        self.response_msg = utils.RESPONSE_MSG[utils.AGE_LIST.index(self.age)][random.randint(0, 3)]

        self.title = Label(text=f"{self.response_title}",
                           font_size=utils.SUBTITLE_FONT_SIZE,
                           font_name=utils.FONT_NAME,
                           size_hint=(0.8, 0.1),
                           pos_hint={'x': 0.1, 'y': 0.6},
                           color=utils.COLOR_WHITE)

        self.response_time_txt = Label(text=f"{self.response_time}ms",
                           font_size=utils.SUBTITLE_FONT_SIZE,
                           font_name=utils.FONT_NAME,
                           size_hint=(0.8, 0.1),
                           pos_hint={'x': 0.1, 'y': 0.6},
                           color=utils.COLOR_WHITE)

        self.description = Label(text=f"당신의 반응속도 나이는 {self.age}",
                                 font_size=utils.DESCRIPTION_FONT_SIZE,
                                 font_name=utils.FONT_NAME,
                                 size_hint=(0.8, 0.2),
                                 pos_hint={'x': 0.1, 'y': 0.3},
                                 color=utils.COLOR_WHITE)

        self.description_msg = Label(text=f"{self.response_msg}",
                                 font_size=utils.DESCRIPTION_FONT_SIZE,
                                 font_name=utils.FONT_NAME,
                                 size_hint=(0.8, 0.2),
                                 pos_hint={'x': 0.1, 'y': 0.3},
                                 color=utils.COLOR_WHITE)

        self.restart_txt = Label(text="PLAY AGAIN?",
                                font_size=utils.DESCRIPTION_FONT_SIZE,
                                font_name=utils.FONT_NAME,
                                size_hint=(0.8, 0.2),
                                pos_hint={'x': 0.1, 'y': 0.1},
                                color=utils.COLOR_WHITE)

        self.restart_button = Label(text="YES",
                                    font_size=utils.DESCRIPTION_FONT_SIZE,
                                    size_hint=(0.2, 0.1),
                                    pos_hint={'x': 0.4, 'y': 0.1})
        self.restart_button.bind(on_press=switch_callback_main)

        self.home_button = Label(text="NO",
                                  font_size=utils.DESCRIPTION_FONT_SIZE,
                                  size_hint=(0.2, 0.1),
                                  pos_hint={'x': 0.4, 'y': 0.1})
        self.restart_button.bind(on_press=switch_callback_game)

        self.add_widget(self.title)
        self.add_widget(self.response_time_txt)
        self.add_widget(self.description)
        self.add_widget(self.restart_txt)
        self.add_widget(self.restart_button)
        self.add_widget(self.home_button)
