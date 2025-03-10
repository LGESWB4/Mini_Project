from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
import utils
import GameSetting as GS

class ResultScreen(FloatLayout):
    def __init__(self, switch_callback, **kwargs):
        super().__init__(**kwargs)

        self.age = 10
        self.total_score = 0
        self.total_reaction_time = 0

        self.title = Label(text="반응 속도",
                           font_size=utils.SUBTITLE_FONT_SIZE,
                           font_name=utils.FONT_NAME,
                           size_hint=(0.8, 0.1),
                           pos_hint={'x': 0.1, 'y': 0.6},
                           color=utils.COLOR_WHITE)
        self.add_widget(self.title)

        self.reaction_time = Label(text=f"{self.total_reaction_time}ms",
                           font_size=utils.SUBTITLE_FONT_SIZE,
                           font_name=utils.FONT_NAME,
                           size_hint=(0.8, 0.1),
                           pos_hint={'x': 0.1, 'y': 0.5},
                           color=utils.COLOR_RED)
        self.add_widget(self.reaction_time)

        self.description = Label(text=f"당신의 반응속도 나이는 {self.age}세",
                                 font_size=utils.DESCRIPTION_FONT_SIZE,
                                 font_name=utils.FONT_NAME,
                                 size_hint=(0.8, 0.2),
                                 pos_hint={'x': 0.1, 'y': 0.3},
                                 color=utils.COLOR_WHITE)
        self.add_widget(self.description)

    def update_results(self,total_score, total_reaction_time):
        self.total_score = total_score
        self.total_reaction_time = total_reaction_time
        self.age, self.score = GS.cal_result(self.total_reaction_time)