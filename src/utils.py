import tflite_runtime.interpreter as tflite
#import tensorflow as tf
import warnings

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
COLOR_WHITE = hex_to_rgb('#E6E6E6')
COLOR_RED = hex_to_rgb('#FF0004')
COLOR_GREEN = hex_to_rgb('#94FA89')
COLOR_BLACK = hex_to_rgb('#000000')
COLOR_NEON_BLUE = hex_to_rgb('#00FFFF')
COLOR_NEON_GREEN = hex_to_rgb('#00FF00')
COLOR_NEON_YELLOW = hex_to_rgb('#FFFF00')
COLOR_PINK = hex_to_rgb('#E14BB6')
COLOR_PURPLE = hex_to_rgb('#5D0865')
COLOR_YELLOW = hex_to_rgb('#FFC000')

# image path
PAPER_IMG = '../etc/images/paper.png'
ROCK_IMG = '../etc/images/rock.png'
SCISSORS_IMG = '../etc/images/scissors.png'
EMOJI_IMG = '../etc/images/emoji.png'
BG_IMG = '../etc/images/main_bg.png'
GAME_IMG = '../etc/images/game_bg.png'

# response time title
AGE_LIST = ['신','10대','20대','30대','40대','50대','60대','70대','실패']
RESPONSE_TIME_TITLE = ["인간 초월자", "초고속 반응", "젊음의 속도", "체감 속도 저하 시작", "노화의 시작", "느림의 미학", "고대 유물급 속도", "유물 속도", "시간 왜곡자"]

# response time message
MSG_GOD = ["당신… 혹시 로봇이세요?", "눈보다 손이 더 빠르네. 반칙 아닌가?", "이거 측정 오류 아님? 관리자 부르세요", "손이 아니라 시간을 조작한 거 아냐?"]
MSG_10 = ["손가락에 터보 엔진 달았냐?", "반응 속도가 아니라 예지력이잖아?", "이 속도면 인류 최후의 희망", "게임 좀 적당히 해라;;"]
MSG_20 = ["젊음의 반응 속도, 아직 남아있네?", "이 정도면 사회생활 잘 버티겠는데?", "20대라며? 왜 10대 반응 속도 나옴?", "슬슬 늙을 준비해라. 한 살 더 먹으면 0.1초 추가됨"]
MSG_30 = ["어? 손가락 굳었어요?", "체력은 괜찮아요? 숨은 안 차죠?", "한 판 할 때마다 스트레칭 필수", "이제 가위바위보도 워밍업 필요함?"]
MSG_40 = ["내가 졌어… 이 속도를 기다릴 인내심이 없거든!", "반응 속도보다 눈 깜빡이는 속도가 더 빠르다", "손 내미는 데 한 세월 걸리네…", "슬슬 거북이랑 대결 준비하자"]
MSG_50 = ["손 내미는 동안 세월이 흐르는 기분", "이 속도면 뇌에서 손끝까지 신호 가는데 왕복 1년 걸릴 듯", "괜찮아요? 손목에 무리 가는 거 아니죠?", "이걸 지켜보는 내 인생이 슬퍼진다…"]
MSG_60 = ["지금 가위 내는 거 맞아요? 아니면 손 올리는 중?", "이 속도면 꿈에서 가위바위보하는 수준", "이거 결과 나올 때까지 연금 수령 가능하겠는데?", "손 내는 속도가 VHS 테이프 배속 0.5배 수준"]
MSG_70 = ["뼈 삐걱거리는 소리 들리는 거 실화냐?", "이거 반응 속도 측정하는 거 맞지? 내 인내력 테스트 아님?", "손 하나 내미는데 인류 역사책 한 권이 지나감", "속도가 아니라 사망 판정 받을 기세"]
MSG_FAIL = ["지금 가위 내는 중이죠? 아니면 명상 중?", "손이 아니라 나무늘보가 반응하는데?", "이 속도면 중간에 밥 먹고 와도 되겠는데?", "이미 상대가 사라졌어요… 아직도 내는 중이세요?", "지금 가위 내는 건데, 혹시 재도전 중 아니죠?"]
RESPONSE_MSG = [MSG_GOD, MSG_10, MSG_20, MSG_30, MSG_40, MSG_50, MSG_60, MSG_70, MSG_FAIL]

warnings.simplefilter("ignore", UserWarning)

# TFLite 모델 로딩
MODEL_PATH = './MobileNetV3A_RP_Hard_LossAll.tflite'
#INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH) # 모델 로딩
INTERPRETER = tflite.Interpreter(model_path=MODEL_PATH) # 모델 로딩
"""try:
    # GPU Delegate 설정
    INTERPRETER = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[tflite.load_delegate('libtensorflowlite_delegate_gpu.so')]
    )
    print("✅ GPU 가속 활성화됨")
except:
    INTERPRETER = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    print("🖥️ CPU로 실행") # 모델 로딩"""
# 카메라 설정
WIDTH, HEIGHT = 400, 300  # 원하는 해상도로 설정

INTERPRETER.allocate_tensors() # tensor 할당
INPUT_DETAILS = INTERPRETER.get_input_details()  # input tensor 정보 얻기
OUTPUT_DETAILS = INTERPRETER.get_output_details() # output tensor 정보 얻기
INPUT_DTYPE = INPUT_DETAILS[0]['dtype']
INPUT_HEIGHT = INPUT_DETAILS[0]['shape'][1]
INPUT_WIDTH = INPUT_DETAILS[0]['shape'][2]


print('model input shape:', (HEIGHT, WIDTH))
#print(INPUT_DETAILS)
#print(OUTPUT_DETAILS)

# Game settings
MODEL_TO_ANS = [1, 0, 2, 3]
CLASS_LIST = '_ Scissors Rock Paper'.split()
ANSTOTEXT = {0: 'Scissors', 1: 'Rock', 2: 'Paper'}
COLORLIST = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

THRESHOLD = 0.65 # 모델 정확도

COUNT_WIN = 5 # 이긴 frame 세팅값 (debouncing)
GAME_ROUND = 3 # 게임 라운드 세팅값

TIMING_LIST = [100, 300, 400, 500, 600, 700, 800, 1000]

AGE_LIST = ['신', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '실패']
SCORE_LIST = [100, 90, 80, 70, 60, 50, 40, 30, 0]