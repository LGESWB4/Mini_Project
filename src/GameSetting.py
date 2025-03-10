import utils


# reaction_time 에 따른 점수 계산
def cal_result(result_time):
    idx = -1

    timing_list = utils.TIMING_LIST
    age_list = utils.AGE_LIST
    score_list = utils.SCORE_LIST

    if result_time == 0:
        idx = 8
    elif result_time <= timing_list[0]:
        idx = 0
    elif timing_list[0] < result_time <= timing_list[1]:
        idx = 1
    elif timing_list[1] < result_time <= timing_list[2]:
        idx = 2
    elif timing_list[2] < result_time <= timing_list[3]:
        idx = 3
    elif timing_list[3] < result_time <= timing_list[4]:
        idx = 4
    elif timing_list[4] < result_time <= timing_list[5]:
        idx = 5
    elif timing_list[5] < result_time <= timing_list[6]:
        idx = 6
    elif timing_list[6] < result_time <= timing_list[7]:
        idx = 7
    elif result_time > timing_list[7]:
        idx = 8

    age = age_list[idx]
    score = score_list[idx]
    return age, score