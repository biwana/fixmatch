def color_to_num(color):
    if color == (200, 200, 200):
        num = 0
    elif color == (255, 0, 0):
        num = 1
    elif color == (255, 255, 0):
        num = 2
    elif color == (0, 255, 0):
        num = 3
    elif color == (0, 255, 255):
        num = 4
    elif color == (0, 0, 255):
        num = 5
    elif color == (255, 0, 255):
        num = 6
    elif color == (128, 0, 0):
        num = 7
    elif color == (128, 128, 0):
        num = 8
    elif color == (0, 128, 0):
        num = 9
    elif color == (0, 0, 128):
        num = 10
    elif color == (64, 64, 64):
        num = 11
    else:
        # sys.exit("invalid color:" + str(color))
        num = -1
    return num


def num_to_color(num):
    if isinstance(num, list):
        num = num[0]

    if num == 0:
        color = (200, 200, 200)
    elif num == 1:
        color = (255, 0, 0)
    elif num == 2:
        color = (255, 255, 0)
    elif num == 3:
        color = (0, 255, 0)
    elif num == 4:
        color = (0, 255, 255)
    elif num == 5:
        color = (0, 0, 255)
    elif num == 6:
        color = (255, 0, 255)
    elif num == 7:
        color = (128, 0, 0)
    elif num == 8:
        color = (128, 128, 0)
    elif num == 9:
        color = (0, 128, 0)
    elif num == 10:
        color = (0, 0, 128)
    elif num == 11:
        color = (64, 64, 64)
    else:
        sys.exit("invalid number:" + str(num))
    return color