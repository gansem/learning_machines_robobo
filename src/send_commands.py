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
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)

    rob.play_simulation()

    # speed of wheels (+ direction)
    l_wheel = 25
    r_wheel = 25
    # time spent moving (ms)
    t_move = 500

    # actual measure to check rewards against (max value)
    v_measure = 0

    # Following code moves the robot (forever)
    while True:
        # print("robobo is at {}".format(rob.position()))
        rob.move(l_wheel, r_wheel, t_move)
        print(f"ROB Irs: {rob.read_irs()}")

        # if any IRS detects an object, add to validity measure
        if any(rob.read_irs()):
            v_measure += np.mean([l_wheel, r_wheel]) * t_move

        
   
    print("robobo is at {}".format(rob.position()))
    rob.sleep(1)

    # Following code moves the phone stand
    # rob.set_phone_pan(343, 100)
    # rob.set_phone_tilt(109, 100)
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    # Following code gets an image from the camera
    # image = rob.get_image_front()
    # cv2.imwrite("test_pictures.png",image)

    # time.sleep(0.1)

    # IR reading
    # for i in range(100):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    #     time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()
    
    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
