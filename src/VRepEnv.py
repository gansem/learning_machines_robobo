import math
import random as rnd

import cv2
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box

import info
import robobo
import vrep


class VRepEnv:
    """Class to plug the VRep simulator environment into the Stable - baseline DQN algorithm."""

    def __init__(self, actions, n_observations):
        """
        :param actions: list of actions. Each action is a four-tuple (left_speed, right_speed, duration, direction(-1=backwards, 1=forward))
        :param n_observations: number of sensors
        """
        self.rob = robobo.SimulationRobobo(info.client).connect(address=info.ip, port=19997)
        # using action and observation spaces of Gym to minimize code alterations.
        self.actions = actions
        self.action_space = Discrete(len(actions))
        self.rob.play_simulation()
        self.prey = robobo.SimulationRoboboPrey().connect(address=info.ip, port=19989)
        # self.prey.run()
        self.rob.set_phone_tilt(np.pi / 4.0, 10)
        self.observations = self.get_camera_observations()
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))
        self.time_passed = 0
        self.df = pd.DataFrame()
        self.accu_reward = 0
        self.episode_counter = 0
        self.food_names = ['Food', 'Food0', 'Food1', 'Food2', 'Food3', 'Food4', 'Food5']  # TOdO: make this generic for arbitraty foods
        self.food_eaten = 0

        # temp
        self.img = []
        self.mask = []

    def reset(self):
        '''
        Done at every episode end
        '''
        # validation measures
        self.accu_reward = 0
        self.time_passed = 0

        return self.observations

    def pred_step(self, action_index, epsilon=0, mode='learning'):
        """
        Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)

        :param action_index: Index of the action that should be executed.
        :param epsilon: Current epsilon (for epsilon greedy; used here just for printing)
        :return: tuple of observed state, observed reward, wheter the episode is done and information (not used here)
        """
        # ----- Performing action
        #old_reward = self.get_pred_reward()
        old_irs = self._get_sensor_observations()
        old_irs = [old_irs[i] for i in [1, 2, 3]]
        obj_in_front = False
        for irs in old_irs:
            if irs < 0.15:
                obj_in_front = True
                break
        old_amount_food = self.food_eaten
        action = self.actions[action_index]
        # perform action in environment
        self.rob.move(action[0], action[1], action[2])
        # save stopping ir readings for relevant sensors
        self.observations = self.get_camera_observations()
        self.time_passed += action[2]
        #self.food_eaten = self.rob.collected_food() - self.episode_counter * len(self.food_names)

        # ------ Calculating reward
        reward = self.get_pred_reward()
        if old_amount_food < self.food_eaten and obj_in_front:  # food must be in front of it and robobo must eat it
            reward += 20
        self.accu_reward += reward

        # ------ printing for debugging
        print('\n---- action:', action_index)
        print('reward:', reward)
        print('elapsed time:', self.time_passed)
        print('collected food:', self.food_eaten)

        # ------ Stopping and resetting
        done = False
        if self.food_eaten == len(self.food_names):
            done = True

        # reset metrics after each episode
        if done:
            print('episode done')
            self.episode_counter += 1
            self.food_eaten = 0
            self._reset_food()
            # save the normalized time in dataframe
            entry = {'avg_food_distance': self.get_avg_food_distance(),
                     'time_passed': self.time_passed,
                     'accu_reward': self.accu_reward}
            self.df = self.df.append(entry, ignore_index=True)

            # write dataframe to disk
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/{mode}_progress.tsv', sep='\t',
                           mode='w+')

        return self.observations, reward, done, {}

    def prey_step(self, action_index, epsilon=0, mode='learning'):
        """
        Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)

        :param action_index: Index of the action that should be executed.
        :param epsilon: Current epsilon (for epsilon greedy; used here just for printing)
        :return: tuple of observed state, observed reward, wheter the episode is done and information (not used here)
        """
        # ----- Performing action
        # old_reward = self.prey_rd()
        old_irs = self._get_sensor_observations()
        old_irs = [old_irs[i] for i in [1, 2, 3]]
        obj_in_front = False
        for irs in old_irs:
            if irs < 0.15:
                obj_in_front = True
                break
        old_amount_food = self.food_eaten
        action = self.actions[action_index]
        # perform action in environment
        self.rob.move(action[0], action[1], action[2])
        # save stopping ir readings for relevant sensors
        self.observations = self.get_camera_observations()
        self.time_passed += action[2]
        # self.food_eaten = self.rob.collected_food() - self.episode_counter * len(self.food_names)

        # ------ Calculating reward
        reward = self.get_prey_reward()
        if old_amount_food < self.food_eaten and obj_in_front:  # food must be in front of it and robobo must eat it
            reward += 20
        self.accu_reward += reward

        # ------ printing for debugging
        print('\n---- action:', action_index)
        print('reward:', reward)
        print('elapsed time:', self.time_passed)
        print('collected food:', self.food_eaten)

        # ------ Stopping and resetting
        done = False
        if self.food_eaten == len(self.food_names):
            done = True

        # reset metrics after each episode
        if done:
            print('episode done')
            self.episode_counter += 1
            self.food_eaten = 0
            self._reset_food()
            # save the normalized time in dataframe
            entry = {'avg_food_distance': self.get_avg_food_distance(),
                     'time_passed': self.time_passed,
                     'accu_reward': self.accu_reward}
            self.df = self.df.append(entry, ignore_index=True)

            # write dataframe to disk
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/{mode}_progress.tsv', sep='\t',
                           mode='w+')

        return self.observations, reward, done, {}

    def get_camera_observations(self, include_sensor=False):
        image = self.rob.get_image_front()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Uncomment for foraging
        # low = np.array([25, 52, 72])
        # high = np.array([102, 255, 255])

        # Uncomment for prey
        low = np.array([0, 50, 20])
        high = np.array([5, 255, 255])
        mask = cv2.inRange(hsv, low, high)

        far_left = mask[:, 0:25]
        #left = mask[:, 0:51]
        mid_left = mask[:, 25:51]
        mid = mask[:, 51:77]
        mid_right = mask[:, 77:103]
        #right = mask[:, 77:]
        far_right = mask[:, 103:]

        cam_values = [far_left, mid_left, mid, mid_right, far_right]
        #cam_values = [left, mid, right]

        cam_obs = [(np.sum(value) / (value.shape[0] * value.shape[1]))/255 for value in cam_values]

        if include_sensor:
            front_sensor = [self._get_sensor_observations()[2]]
            observation = [front_sensor.append(cam_ob) for cam_ob in cam_obs]
        else:
            observation = cam_obs

        self.img = image
        self.mask = mask

        return observation

    def _get_sensor_observations(self):
        '''
        Reads sensor information and returns normalized an thresholded values.
        '''
        observation = self.rob.read_irs()
        observation = [observation[i] for i in [1, 3, 5, 7]]  # reading only sensors: backC, frontRR,  frontC, frontLL
        observation = [0.15 if observation[i] == False else observation[i] for i in
                       range(len(observation))]  # false -> 0.15
        # we need to introduce a threshold s.t. only distances below 0.15 are counted. Otherwise the distances
        # will OFTEN be triggered when just moving in an empy plane because of the tilt of the robot.
        observation = [0.15 if observation[i] > 0.15 else observation[i] for i in range(len(observation))]
        return observation

    def get_pred_reward(self):
        _obs = self.observations

        # find mid in observation list
        mid_index = math.floor(len(_obs)/2)
        # find the most food in image section
        max_index = _obs.index(max(_obs))

        # if max is in left or right, apply negative reward according to distance
        if mid_index == max_index:
            reward = _obs[max_index]
        # if max is in mid, apply positive reward according to distance
        else:
            # if 5 observations, reduce -ve reward to mid_left and mid_right
            if len(_obs) > 3:
                if mid_index-1 == max_index or mid_index+1 == max_index:
                    reward = _obs[max_index]-0.7
                else:
                    reward = _obs[max_index]-1
            else:    
                reward = _obs[max_index]-1

        return reward

    def get_prey_reward(self):
        _obs = self.observations

        # find mid in observation list
        mid_index = math.floor(len(_obs)/2)
        # find the most food in image section
        max_index = _obs.index(max(_obs))

        # if max is in left or right, apply negative reward according to distance
        if mid_index == max_index:
            reward = _obs[max_index]
        # if max is in mid, apply positive reward according to distance
        else:
            # if 5 observations, reduce -ve reward to mid_left and mid_right
            if len(_obs) > 3:
                if mid_index-1 == max_index or mid_index+1 == max_index:
                    reward = _obs[max_index]-0.7
                else:
                    reward = _obs[max_index]-1
            else:
                reward = _obs[max_index]-1

        return reward

    def get_avg_food_distance(self):
        '''
        Calculates the average distance of the food objects in the scene. Assuming that there are always 7 food objects.
        :return: Average distance of food objects
        '''
        food_positions = []
        # collect positions of all the foods
        for food in self.food_names:
            food_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(self.rob._clientID, food, vrep.simx_opmode_blocking))
            food_position = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self.rob._clientID, food_handle, -1, vrep.simx_opmode_blocking))
            food_positions.append(food_position)

        food_positions = np.array(food_positions)

        # compute all distances in the 2D plane
        distances = []
        for i in range(len(food_positions)):
            for j in range(i+1, len(food_positions)):
                distances.append(math.sqrt((food_positions[i][0] - food_positions[j][0])**2
                                           + (food_positions[i][1] - food_positions[j][1])**2))

        return np.array(distances).mean()

    def _reset_food(self):
        #collects positions of the walls (only works on standard map)
        leftwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(self.rob._clientID, '80cmHighWall200cm', vrep.simx_opmode_blocking))
        pos_lw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self.rob._clientID, leftwall_handle, -1, vrep.simx_opmode_blocking))
        
        rightwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(self.rob._clientID, '80cmHighWall200cm1', vrep.simx_opmode_blocking))
        pos_rw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self.rob._clientID, rightwall_handle, -1, vrep.simx_opmode_blocking))
        
        bottomwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(self.rob._clientID, '80cmHighWall200cm2', vrep.simx_opmode_blocking))
        pos_bw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self.rob._clientID, bottomwall_handle, -1, vrep.simx_opmode_blocking))
        
        topwall_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(self.rob._clientID, '80cmHighWall200cm0', vrep.simx_opmode_blocking))
        pos_tw = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self.rob._clientID, topwall_handle, -1, vrep.simx_opmode_blocking))
        
        food_pos = []
        for food in self.food_names:
            food_handle = vrep.unwrap_vrep(vrep.simxGetObjectHandle(self.rob._clientID, food, vrep.simx_opmode_blocking))
            new_pos = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self.rob._clientID, food_handle, -1, vrep.simx_opmode_blocking))
            if new_pos[2] - 1 > 0:
                new_pos[2] -= 1  # reset height of food
            # from provided script on canvas
            new_pos[0] = (rnd.uniform((pos_lw[0] + 0.25), (pos_rw[0] - 0.25)))
            new_pos[1] = (rnd.uniform((pos_bw[1] + 0.25), (pos_tw[1] - 0.25)))
            # check if it is placed on the robot
            while self.rob.position()[0] + 0.25 > new_pos[0] > self.rob.position()[0] - 0.25 \
                    and self.rob.position()[1] + 0.25 > new_pos[1] > self.rob.position()[1] - 0.25:
                new_pos[0] = (rnd.uniform((pos_lw[0] + 0.25), (pos_rw[0] - 0.25)))
                new_pos[1] = (rnd.uniform((pos_bw[1] + 0.25), (pos_tw[1] - 0.25)))
                #checks if the food doesn't overlap already placed food
                for x in food_pos:
                    if new_pos == x:
                        new_pos[0] = (rnd.uniform((pos_lw[0] + 0.25), (pos_rw[0] - 0.25)))
                        new_pos[1] = (rnd.uniform((pos_bw[1] + 0.25), (pos_tw[1] - 0.25)))      
                    else:
                        food_pos.append(new_pos)
            vrep.simxSetObjectPosition(self.rob._clientID, food_handle, -1, new_pos, vrep.simx_opmode_blocking)
