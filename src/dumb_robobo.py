#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np
import pandas as pd
import info

import robobo
import cv2
import sys
import signal
import prey


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def mean(arr):
    'return mean of array in type float64'
    return np.mean(arr, dtype=np.float64)

def decide_move(irs, actions):
    'decide on actions to take'
    # [backR (0), backC (1), backL (2), frontRR (3), frontR (4), frontC (5), frontL (6), frontLL (7)]]
    # if there is on obstacle detected, go straight
    if np.mean(irs[4:]) > 0.8:
        l, r, t = actions[0]
    # if there is something closer in front than on the back, go forward
    elif (mean(irs[4:])*3)/4 > 0.05:
        # if there is something closest straight in front, spin right or left
        if irs[5] < mean([irs[3],irs[4]]) or irs[5] < mean([irs[6],irs[7]]):
            # if right is closest than left, spin left:
            if mean([irs[5],irs[6],irs[7]]) > mean([irs[3],irs[4],irs[5]]):
                l, r, t = actions[3]
            # spin right
            else:
                l, r, t = actions[4]
        else:
            # if right is closest than left, turn left:
            if mean([irs[5],irs[6],irs[7]]) > mean([irs[3],irs[4],irs[5]]):
                l, r, t = actions[1]
            # turn right
            else:
                l, r, t = actions[2]
    # go back
    else:
        # if right is closer than left, turn back-left
        if mean([irs[2],irs[1]]) > mean([irs[0],irs[1]]):
            l, r, t = actions[5]
        # turn back-right
        else:
            l, r, t = actions[6]
    return l, r, t

def decide_move_reduced_actions(irs, actions):
    'decide on actions to take while using reduced actions (forward, rotate-left, rotate-right)'

    object_in_range = False
    # [backR (0), backC (1), backL (2), frontRR (3), frontR (4), frontC (5), frontL (6), frontLL (7)]]
    # if there is on obstacle detected, go straight
    if np.mean(irs[4:]) > 0.78:
        i = 0
    # if there is something closer in front than on the back, go forward
    else:
        object_in_range = True
        # if right is closest than left, turn left:
        if mean([irs[5],irs[6],irs[7]]) > mean([irs[3],irs[4],irs[5]]):
            i = 2
        # turn right
        else:
            i = 1
    return i, object_in_range


def main():
    """ obstacle avoidance hard-coded algorithm """

    # allow for termination of process
    signal.signal(signal.SIGINT, terminate_program)

    # connect to environment and start simulation
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
    rob.play_simulation()

    # import actions
    actions = info.actions
    # if actions are reduced (similar to smart robobo), remove reward element
    if len(actions[0]) > 3:
        for i in range(len(actions)):
            actions[i] = actions[i][0:3]

    # init vars
    time_passed = 0
    df = pd.DataFrame()


    # loop forever until terminated manually
    while True:
        # convert False to 0s
        irs = np.array(rob.read_irs(), dtype=np.float64)
        # convert 0s to 1s
        irs_temp = irs
        for i in range(len(irs)):
            if (irs==0)[i]:
                irs_temp[i] = 1
        irs = irs_temp

        action_index, object_in_range = decide_move_reduced_actions(irs, actions)
        action = actions[action_index]
        l, r, t = action

        # print actions (l_wheel speed | r_wheel speed | time moving (ms))
        print(f'action: {action_index}:\t{l}\t{r}\t{t}')

        # move robot
        rob.move(l, r, t)

        df = df.append({
            "action_index": action_index,
            # "episode_index": int(self.episode_counter),
            # "observations:": self.observations,
            # "reward": overall_reward,
            "object_in_range": object_in_range,
            # TODO: v_measure_calc_distance in  VRepEnv has a mistake! action[3] = 1.. should be action[2]
            "v_measure_calc_distance": (1 * np.mean([action[0], action[1]]))/10,
            "v_measure_sensor_distance": np.sum([(0.2-x) for x in rob.read_irs()]),
            # "v_distance_reward": distance,
            # "accu_v_measure_sensor_distance": self.accu_v_measure_sensor_distance,
            # "accu_reward": self.accu_reward,
            # "epsilon": epsilon,
            }, ignore_index=True)

        time_passed += t
        # if 5 minutes passed, print save dataframe and stop
        if time_passed > 240000:
            df.to_csv(f'results/{info.task}/{info.user}/{info.scene}/dumb_robobo_progress.tsv', sep='\t')
            break



if __name__ == "__main__":
    main()
