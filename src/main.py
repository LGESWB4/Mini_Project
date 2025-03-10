from kivy.app import App
from kivy.core.window import Window
from MainScreen import MainScreen
from GameScreen import GameScreen
from ResultScreen import ResultScreen
import os

os.environ['NUMEXPR_MAX_THREADS'] = '1'

Window.size = (1280, 720)

class RockPaperScissorsApp(App):
    def build(self):
        self.main_screen = MainScreen(self.switch_to_game)
        self.game_screen = GameScreen(self.switch_to_result)
        self.result_screen = ResultScreen(self.switch_to_main)

        self.current_screen = self.main_screen
        return self.current_screen

    def switch_to_main(self, instance):
        self.game_screen.stop_camera()
        self.current_screen = self.main_screen
        self.root.clear_widgets()
        self.root.add_widget(self.main_screen)

    def switch_to_game(self, instance):
        self.current_screen = self.game_screen
        self.root.clear_widgets()
        self.root.add_widget(self.game_screen)

    def switch_to_result(self, total_score, total_reaction_time):
        self.result_screen.update_results(total_score, total_reaction_time)
        self.current_screen = self.result_screen
        self.root.clear_widgets()
        self.root.add_widget(self.result_screen)



if __name__ == "__main__":
    RockPaperScissorsApp().run()