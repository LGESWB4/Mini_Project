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