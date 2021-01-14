#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():

    def mean(arr):
        return np.mean(arr, dtype=np.float64)

    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
    rob.play_simulation()

    time_step = 500

    actions = [
        [35, 35, time_step],       # forward (0)
        [10, 35, time_step],        # turn left (1)
        [35, 10, time_step],        # turn right (2)
        [-10, 10, time_step],      # spin left (3)
        [10, -10, time_step],      # spin right (4)
        [-20, -10, time_step],     # backwards--turn-left (5)
        [-10, -20, time_step],     # backwards--turn-right (6)
    ]

    while True:
        # convert False to 0s
        irs = np.array(rob.read_irs(), dtype=np.float64)
        # convert 0s to 1s
        irs_temp = irs
        for i in range(len(irs)):
            if (irs==0)[i]:
                irs_temp[i] = 1
        irs = irs_temp


        #  [backR (0), backC (1), backL (2), frontRR (3), frontR (4), frontC (5), frontL (6), frontLL (7)]

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

        print(l, r, t)
        rob.move(l, r, t)

if __name__ == "__main__":
    main()
