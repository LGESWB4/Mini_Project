from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics import Line, Color
import utils

class MainScreen(FloatLayout):
    def __init__(self, switch_callback, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            Color(rgba=utils.COLOR_BLUE)
            self.border_line = Line(rectangle=(50, 50, self.width - 100, self.height - 100), dash_offset=2, dash_length=5)

        self.switch_callback = switch_callback
        self.bind(size=self.update_border, pos=self.update_border)

        self.title = Label(text="덕륜이를 이겨라",
                           font_size=utils.TITLE_FONT_SIZE,
                           font_name=utils.FONT_NAME,
                           size_hint=(0.8, 0.1),
                           pos_hint={'x': 0.1, 'y': 0.6},
                           color=utils.COLOR_BLUE,
                           outline_color=utils.COLOR_BLUE_DARK,
                           outline_width=5)
        self.add_widget(self.title)

        self.description = Label(text="도박에 중독된 덕륜.. 그는 가위바위보에 묵숨을 걸었는데...\n 과연 그를 이길 수 있는가...", font_size=utils.DESCRIPTION_FONT_SIZE, font_name=utils.FONT_NAME, size_hint=(0.8, 0.2), pos_hint={'x': 0.1, 'y': 0.45})
        self.add_widget(self.description)

        self.start_button = Button(background_normal = '../etc/images/start.png', size_hint=(0.2, 0.1), pos_hint={'x': 0.35, 'y': 0.35})
        self.start_button.bind(on_press=self.switch_callback)
        self.add_widget(self.start_button)

    def update_border(self, *args):
        self.border_line.rectangle = (50, 50, self.width - 100, self.height - 100)

    def next_screen(self):
        print('next screen')
        #self.switch_callback()