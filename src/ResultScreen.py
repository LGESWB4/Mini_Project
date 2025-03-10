from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.animation import Animation
import utils
import random
import GameSetting as GS

class ResultScreen(FloatLayout):
    def __init__(self, switch_callback_main, switch_callback_game, **kwargs):
        super().__init__(**kwargs)

        self.age = '10대'
        self.response_time = 0
        self.response_title = utils.RESPONSE_TIME_TITLE[utils.AGE_LIST.index(self.age)]
        self.response_msg = utils.RESPONSE_MSG[utils.AGE_LIST.index(self.age)][random.randint(0, 3)]

        self.bg = Image(source=utils.BG_IMG, allow_stretch=True, keep_ratio=False)
        self.add_widget(self.bg)

        self.title = Label(text=f"{self.response_title}",
                           font_size=utils.DESCRIPTION_FONT_SIZE,
                           font_name=utils.FONT_NAME,
                           markup=True,
                           size_hint=(0.8, 0.1),
                           pos_hint={'center_x': 0.5, 'center_y': 0.8},
                           color=utils.COLOR_WHITE)
        self.add_widget(self.title)

        self.response_time_txt = Label(text=f"{self.response_time}ms",
                           font_size=utils.TITLE_FONT_SIZE * 1.5,
                           font_name=utils.FONT_NAME,
                           size_hint=(0.8, 0.15),
                           pos_hint={'center_x': 0.5, 'center_y': 0.65},
                           color=utils.COLOR_GREEN)
        self.add_widget(self.response_time_txt)

        self.description = Label(text=f"당신의 반응속도 나이는 [color=#A65CFF]{self.age}[/color]",
                                 font_size=utils.SUBTITLE_FONT_SIZE,
                                 font_name=utils.FONT_NAME,
                                 size_hint=(0.8, 0.1),
                                 markup=True,
                                 pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                 color=utils.COLOR_WHITE)
        self.add_widget(self.description)

        self.description_msg = Label(text=f"{self.response_msg}",
                                 font_size=utils.DESCRIPTION_FONT_SIZE,
                                 font_name=utils.FONT_NAME,
                                 size_hint=(0.8, 0.1),
                                 pos_hint={'center_x': 0.5, 'center_y': 0.4},
                                 color=utils.COLOR_RED)
        self.add_widget(self.description_msg)

        self.restart_txt = Label(text="PLAY AGAIN?",
                                font_size=utils.DESCRIPTION_FONT_SIZE,
                                font_name=utils.FONT_NAME,
                                size_hint=(0.8, 0.1),
                                pos_hint={'center_x': 0.5, 'center_y': 0.3},
                                color=utils.COLOR_WHITE)
        self.add_widget(self.restart_txt)

         # 버튼 박스
        self.button_container = BoxLayout(orientation='horizontal',
                                          size_hint=(0.3, 0.1),
                                          pos_hint={'center_x': 0.5, 'center_y': 0.18},
                                          spacing=10)

        # "YES" 버튼
        self.restart_button = Button(text="▶ YES",
                                     font_size=utils.DESCRIPTION_FONT_SIZE,
                                     font_name=utils.FONT_NAME,
                                     background_color=(0, 0, 0, 1),
                                     color=(1, 1, 1, 1),
                                     size_hint=(0.5, 1))
        self.restart_button.bind(on_press=switch_callback_game)
        self.button_container.add_widget(self.restart_button)

        # "NO" 버튼
        self.home_button = Button(text="▶ NO",
                                  font_size=utils.DESCRIPTION_FONT_SIZE,
                                  font_name=utils.FONT_NAME,
                                  background_color=(0, 0, 0, 1),
                                  color=(1, 1, 1, 1),
                                  size_hint=(0.5, 1))
        self.home_button.bind(on_press=switch_callback_main)
        self.button_container.add_widget(self.home_button)

        self.add_widget(self.button_container)

        self.emoji = Image(
            source='../etc/images/emoji.png',
            size_hint=(None, None),
            size=(100, 100),
            pos_hint={'center_x': 0.9, 'y': 0.44})
        self.add_widget(self.emoji)

        # 애니메이션 실행
        self.start_shaking_animation()

    def start_shaking_animation(self):
        anim = Animation(pos_hint={'center_x': 0.91}, duration=0.5) + \
               Animation(pos_hint={'center_x': 0.9}, duration=0.5)
        anim.repeat = True
        anim.start(self.emoji)

    def update_results(self, total_score, response_time):
        self.total_score = total_score
        self.response_time = response_time
        self.age, self.score = GS.cal_result(self.response_time)
        self.response_title = utils.RESPONSE_TIME_TITLE[utils.AGE_LIST.index(self.age)]
        self.response_msg = utils.RESPONSE_MSG[utils.AGE_LIST.index(self.age)][random.randint(0, 3)]

        self.title.text = f"{self.response_title}"
        self.response_time_txt.text = f"{self.response_time:.2f}ms"
        self.description.text = f"당신의 반응속도 나이는 [color=#A65CFF]{self.age}[/color]"
        self.description_msg.text= f"{self.response_msg}"