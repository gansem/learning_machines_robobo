#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np
import pandas as pd
import math
import random
import info

import robobo
import cv2
import sys
import signal
import prey
import vrep

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def mean(arr):
    'return mean of array in type float64'
    return np.mean(arr, dtype=np.float64)

def decide_irs_move(irs, actions):
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

def decide_irs_move_reduced_actions(irs, actions):
    'decide on actions to take while using reduced actions (forward [0], rotate-left [1], rotate-right [2])'

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

def decide_cam_move_reduced_actions(cam_obs, actions):
    'decide on actions to take based on camera observations using reduced actions (forward [0], rotate-left [1], rotate-right [2])'

    # find mid in observation list
    mid_index = math.floor(len(cam_obs) / 2)
    # find the most food in image section (middle has increased weighting)
    obs_values = []
    for i, v in enumerate(cam_obs):
        value = v
        if mid_index == i:
            value = v*1.5
        obs_values.append(value)
    max_index = obs_values.index(max(obs_values))

    print(f"observations values:\t{obs_values[0]}\t{obs_values[1]}\t{obs_values[2]}")

    # if food is in the middle, go straight forward
    if max_index == mid_index:
        i = 0
    # if food is on the right, turn right
    elif max_index > mid_index:
        i = 1
    # otherwise, turn left
    else:
        i = 2        

    return i

def get_image_segments(rob, search_food):
    '''
    Get image from front robot camera,
    apply mask to identify food / prey,
    get percentages of visibility per segment
    '''
    image = rob.get_image_front()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # if searching for food, look for green colours
    if search_food:
        low = np.array([25, 52, 72])
        high = np.array([102, 255, 255])
    # else, prey
    else:
        pass
    mask = cv2.inRange(hsv, low, high)

    # far_left = mask[:, 0:23]
    # mid_left = mask[:, 23:46]
    left = mask[:, 0:40]
    # mid = mask[:, 46:82]
    mid = mask[:, 40:88]
    # mid_right = mask[:, 82:105]
    right = mask[:, 88:]
    # far_right = mask[:, 105:]

    # cam_values = [far_left, mid_left, mid, mid_right, far_right]
    cam_values = [left, mid, right]

    cam_obs = [(np.sum(value) / (value.shape[0] * value.shape[1]))/255 for value in cam_values]

    # if include_sensor:
    #     front_sensor = [self._get_sensor_observations()[2]]
    #     observation = [front_sensor.append(cam_ob) for cam_ob in cam_obs]
    # else:
    observation = cam_obs

    return observation


def main(task):
    # allow for termination of process
    signal.signal(signal.SIGINT, terminate_program)
    # connect to environment and start simulation
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
    rob.play_simulation()
    # tilt robot camera a bit downwards
    rob.set_phone_tilt(np.pi / 4.0, 10)

    # import actions
    actions = info.actions
    # if actions are reduced (similar to smart robobo), remove reward element
    if len(actions[0]) > 3:
        for i in range(len(actions)):
            actions[i] = actions[i][0:3]

    # choose between tasks
    if task==1:
        object_avoidance(rob, actions)
    elif task==2:
        foraging(rob, actions)
    # elif task==3:
    #     chasing_prey(rob, actions)

def parse_irs(irs_values):
    # convert False to 0s
    irs = np.array(irs_values, dtype=np.float64)
    # convert 0s to 1s
    irs_temp = irs
    for i in range(len(irs)):
        if (irs==0)[i]:
            irs_temp[i] = 1
    return irs_temp

def calc_distance_travelled(start_pos, stop_pos):
    return math.sqrt((stop_pos[0] - start_pos[0])**2 + (stop_pos[1] - start_pos[1])**2)

def _reset_food(rob, food_names):
    # collects positions of the walls (only works on standard map)
    leftwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(rob._clientID, '80cmHighWall200cm', vrep.simx_opmode_blocking))
    pos_lw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(rob._clientID, leftwall_handle, -1, vrep.simx_opmode_blocking))
    
    rightwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(rob._clientID, '80cmHighWall200cm1', vrep.simx_opmode_blocking))
    pos_rw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(rob._clientID, rightwall_handle, -1, vrep.simx_opmode_blocking))
    
    bottomwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(rob._clientID, '80cmHighWall200cm2', vrep.simx_opmode_blocking))
    pos_bw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(rob._clientID, bottomwall_handle, -1, vrep.simx_opmode_blocking))
    
    topwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(rob._clientID, '80cmHighWall200cm0', vrep.simx_opmode_blocking))
    pos_tw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(rob._clientID, topwall_handle, -1, vrep.simx_opmode_blocking))
    
    food_pos = []
    for food in food_names:
        food_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(rob._clientID, food, vrep.simx_opmode_blocking))
        new_pos = vrep.unwrap_vrep(vrep.simxGetObjectPosition(rob._clientID, food_handle, -1, vrep.simx_opmode_blocking))
        if new_pos[2] - 1 > 0:
            new_pos[2] -= 1  # reset height of food
        # from provided script on canvas
        new_pos[0] = (random.uniform((pos_lw[0] + 0.25), (pos_rw[0] - 0.25)))
        new_pos[1] = (random.uniform((pos_bw[1] + 0.25), (pos_tw[1] - 0.25)))
        # check if it is placed on the robot
        while rob.position()[0] + 0.25 > new_pos[0] > rob.position()[0] - 0.25 \
                and rob.position()[1] + 0.25 > new_pos[1] > rob.position()[1] - 0.25:
            new_pos[0] = (random.uniform((pos_lw[0] + 0.25), (pos_rw[0] - 0.25)))
            new_pos[1] = (random.uniform((pos_bw[1] + 0.25), (pos_tw[1] - 0.25)))
            #checks if the food doesn't overlap already placed food
            for x in food_pos:
                if new_pos == x:
                    new_pos[0] = (random.uniform((pos_lw[0] + 0.25), (pos_rw[0] - 0.25)))
                    new_pos[1] = (random.uniform((pos_bw[1] + 0.25), (pos_tw[1] - 0.25)))      
                else:
                    food_pos.append(new_pos)
        vrep.simxSetObjectPosition(rob._clientID, food_handle, -1, new_pos, vrep.simx_opmode_blocking)


def foraging(rob, actions):
    """look for food and eat it"""
    eval_df = pd.DataFrame()
    n_samples = 50
    # init vars
    episode_counter = 0
    time_passed = 0
    df = pd.DataFrame()
    food_eaten = 0
    food_names = ['Food', 'Food0', 'Food1', 'Food2', 'Food3', 'Food4', 'Food5']  # TODO: make this generic for arbitraty foods

    # loop forever until terminated manually
    i = 0
    while i < n_samples:
        irs = parse_irs(rob.read_irs())
        cam_obs = get_image_segments(rob, search_food=True)

        action_index = decide_cam_move_reduced_actions(cam_obs, actions)
        action = actions[action_index]
        l, r, t = action
        
        # save starting positionn (before move)
        start_pos = rob.position()
        # move robot
        rob.move(l, r, t)
        food_eaten = rob.collected_food() - episode_counter * len(food_names)
        # save finishing position (after move)
        stop_pos = rob.position()

        distance = calc_distance_travelled(start_pos, stop_pos)


        # print actions (l_wheel speed | r_wheel speed | time moving (ms))
        print(f'action: {action_index}:\t{l}\t{r}\t{t}\t collected food: {food_eaten}')

        df = df.append({
            "action_index": action_index,
            "time_elapsed": time_passed,
            "episode_index": int(episode_counter),
            # "observations:": self.observations,
            # "reward": overall_reward,
            # "object_in_range": object_in_range,
            # TODO: v_measure_calc_distance in  VRepEnv has a mistake! action[3] = 1.. should be action[2]
            # "v_measure_calc_distance": (1 * np.mean([action[0], action[1]]))/10,
            # "v_measure_sensor_distance": np.sum([(0.2-x) for x in rob.read_irs()]),
            "v_distance_reward": distance,
            # "accu_v_measure_sensor_distance": self.accu_v_measure_sensor_distance,
            # "accu_reward": self.accu_reward,
            # "epsilon": epsilon,
            }, ignore_index=True)

        time_passed += t

        # stopping and resetting
        done = False
        if food_eaten == len(food_names):
            i += 1
            eval_df = eval_df.append({
            "action_index": action_index,
            "time_elapsed": time_passed,
            "episode_index": int(episode_counter)}, ignore_index=True)

            eval_df.to_csv(f'./results/foraging/evaluation/eval_dumb.tsv', sep = '\t')
            done = True
            food_eaten = 0
            episode_counter += 1
            _reset_food(rob, food_names)

        # if 5 minutes passed, print save dataframe and stop
        if time_passed > 240000 or done:
            pass
            #df.to_csv(f'./results/foraging/evaluation/eval_dumb.tsv', sep='\t')
            # break

def object_avoidance(rob, actions):
    """ obstacle avoidance hard-coded algorithm """
    # init vars
    time_passed = 0
    df = pd.DataFrame()

    # loop forever until terminated manually
    while True:

        irs = parse_irs(rob.read_irs())

        action_index, object_in_range = decide_irs_move_reduced_actions(irs, actions)
        action = actions[action_index]
        l, r, t = action

        # print actions (l_wheel speed | r_wheel speed | time moving (ms))
        print(f'action: {action_index}:\t{l}\t{r}\t{t}')
        # save starting positionn (before move)
        start_pos = rob.position()
        # move robot
        rob.move(l, r, t)
        # save finishing position (after move)
        stop_pos = rob.position()

        distance = calc_distance_travelled(start_pos, stop_pos)

        df = df.append({
            "action_index": action_index,
            "time_elapsed": time_passed,
            # "episode_index": int(self.episode_counter),
            # "observations:": self.observations,
            # "reward": overall_reward,
            "object_in_range": object_in_range,
            # TODO: v_measure_calc_distance in  VRepEnv has a mistake! action[3] = 1.. should be action[2]
            "v_measure_calc_distance": (1 * np.mean([action[0], action[1]]))/10,
            "v_measure_sensor_distance": np.sum([(0.2-x) for x in rob.read_irs()]),
            "v_distance_reward": distance,
            # "accu_v_measure_sensor_distance": self.accu_v_measure_sensor_distance,
            # "accu_reward": self.accu_reward,
            # "epsilon": epsilon,
            }, ignore_index=True)

        time_passed += t
        # if 5 minutes passed, print save dataframe and stop
        if time_passed > 240000:
            df.to_csv(f'results/{info.task}/{info.user}/{info.scene}/dumb_robobo_progress.tsv', sep='\t')
            break

# set task: (1) object avoidance, (2) foraging, (3) chasing prey
if __name__ == "__main__":
    main(2)
