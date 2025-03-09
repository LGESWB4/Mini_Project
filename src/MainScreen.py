from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.animation import Animation
import utils

class MainScreen(FloatLayout):
    def __init__(self, switch_callback, **kwargs):
        super().__init__(**kwargs)

        # 배경 이미지 추가
        self.bg = Image(source=utils.BG_IMG, allow_stretch=True, keep_ratio=False)
        self.add_widget(self.bg)

        self.emoji_1 = Image(
            source='../etc/images/emoji1.png',
            size_hint=(None, None),
            size=(100, 100),
            pos_hint={'center_x': 0.7, 'y': 0.15})
        self.add_widget(self.emoji_1)

        self.emoji_2 = Image(
            source='../etc/images/emoji2.png',
            size_hint=(None, None),
            size=(100, 100),
            pos_hint={'center_x': 0.8, 'y': 0.15})
        self.add_widget(self.emoji_2)

        # 애니메이션 실행
        self.start_shaking_animation()

        self.title = Label(text="덕륜이를 이겨라",
                           font_size=utils.TITLE_FONT_SIZE,
                           font_name=utils.FONT_NAME,
                           size_hint = (0.8, 0.2),
                           pos_hint={'center_x': 0.5, 'y': 0.55},  # 정확한 중앙 정렬
                           color=utils.COLOR_PINK,
                           outline_color=utils.COLOR_BLACK,
                           outline_width=3)
        self.add_widget(self.title)

        self.description = Label(text="도박에 중독된 덕륜.. 그는 가위바위보에 묵숨을 걸었는데...\n 과연 그를 이길 수 있는가...",
                                font_size=utils.DESCRIPTION_FONT_SIZE,
                                font_name=utils.FONT_NAME,
                                size_hint = (0.9, 0.1),
                                pos_hint={'center_x': 0.5, 'y': 0.45},  # 중앙 정렬
                                halign='center',  # 가로 중앙 정렬
                                valign='middle',  # 세로 중앙 정렬
                                text_size=(900, None),  # 텍스트 크기 자동 조절
                                )
        self.add_widget(self.description)

        self.start_button = Button(background_normal='../etc/images/start.png',
                                    size_hint = (0.15, 0.1),
                                    pos_hint={'center_x': 0.5, 'y': 0.34}  # 중앙 정렬
        )
        self.start_button.bind(on_press=switch_callback)
        self.add_widget(self.start_button)

    def start_shaking_animation(self):
        anim = Animation(pos_hint={'center_x': 0.70}, duration=0.5) + \
               Animation(pos_hint={'center_x': 0.69}, duration=0.5)
        anim.repeat = True
        anim.start(self.emoji_1)

        anim1 = Animation(pos_hint={'center_x': 0.8}, duration=0.3) + \
               Animation(pos_hint={'center_x': 0.79}, duration=0.3)
        anim1.repeat = True
        anim1.start(self.emoji_2)
