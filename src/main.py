from kivy.app import App
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from MainScreen import MainScreen
from GameScreen import GameScreen
from ResultScreen import ResultScreen

Window.size = (1280, 720)

class RockPaperScissorsApp(App):
    def build(self):
        self.main_screen = MainScreen(self.switch_to_game)
        self.game_screen = GameScreen(self.switch_to_result)
        self.result_screen = ResultScreen(self.switch_to_main, self.switch_to_game)

        self.root = FloatLayout()
        self.root.add_widget(self.main_screen)

        return self.root

    def switch_to_main(self, instance):
        self.root.clear_widgets()
        self.root.add_widget(self.main_screen)

    def switch_to_game(self, instance):
        self.root.clear_widgets()
        self.root.add_widget(self.game_screen)
        self.game_screen.start_game()

    def switch_to_result(self, instance):
        self.game_screen.stop_camera()
        self.root.clear_widgets()
        self.root.add_widget(self.result_screen)

if __name__ == "__main__":
    RockPaperScissorsApp().run()
