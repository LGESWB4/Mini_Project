import tflite_runtime.interpreter as tflite
#import tensorflow as tf

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# Font settings
FONT_NAME = '../etc/fonts/neodgm_code.ttf'
TITLE_FONT_SIZE = 100
SUBTITLE_FONT_SIZE = 50
DESCRIPTION_FONT_SIZE = 30

# Font colors
COLOR_BLUE = hex_to_rgb('#89DDFA')
COLOR_BLUE_DARK = hex_to_rgb('#496675')
COLOR_WHITE = hex_to_rgb('#FFFFFF')
COLOR_RED = hex_to_rgb('#FF0004')

# image path
PAPER_IMG = '../etc/images/paper.png'
ROCK_IMG = '../etc/images/rock.png'
SCISSORS_IMG = '../etc/images/scissors.png'

# TFLite 모델 로딩
MODEL_PATH = 'RPS_PreTrained_SSD.tflite'
INTERPRETER = tflite.Interpreter(model_path=MODEL_PATH) # 모델 로딩
#INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH) # 모델 로딩
INTERPRETER.allocate_tensors() # tensor 할당
INPUT_DETAILS = INTERPRETER.get_input_details()  # input tensor 정보 얻기
OUTPUT_DETAILS = INTERPRETER.get_output_details() # output tensor 정보 얻기
INPUT_DTYPE = INPUT_DETAILS[0]['dtype']
HEIGHT = INPUT_DETAILS[0]['shape'][1]
WIDTH = INPUT_DETAILS[0]['shape'][2]
print('model input shape:', (HEIGHT, WIDTH))
#print(INPUT_DETAILS)
#print(OUTPUT_DETAILS)

# Game settings
CLASS_LIST = '_ Scissors Rock Paper'.split()
COLOR_LIST = [(), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

IMG_SIZE = 320
THRESHOLD = 0.65 # 모델 정확도

COUNT_WIN = 5 # 이긴 frame 세팅값 (debouncing)
GAME_ROUND = 10 # 게임 라운드 세팅값

TIMING_LIST = [100, 300, 400, 500, 600, 700, 800, 1000]

AGE_LIST = ['신', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '실패']
SCORE_LIST = [100, 90, 80, 70, 60, 50, 40, 30, 0]