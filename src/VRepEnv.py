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

    def __init__(self, pred, actions, n_observations, prey=None):
        """
        :param actions: list of actions. Each action is a four-tuple (left_speed, right_speed, duration, direction(-1=backwards, 1=forward))
        :param n_observations: number of sensors
        :param prey: is prey connected (cannot be None)
        """
        self.pred = pred
        # using action and observation spaces of Gym to minimize code alterations.
        self.actions = actions
        self.action_space = Discrete(len(actions))
        self.prey = prey
        self.pred.set_phone_tilt(np.pi / 3.75, 10)
        self.pred_observations = self.get_camera_observations() + [0.0]*6
        self.prey_observations = self._get_sensor_observations()
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))
        self.time_passed = 0
        self.df = pd.DataFrame()
        self.accu_reward = 0
        self.episode_counter = 0
        self.winner = []

        # temp
        self.img = []
        self.mask = []

    def reset(self, role):
        '''
        Done at every episode end
        :param role: 'pred' or 'prey'
        '''
        # validation measures
        self.accu_reward = 0
        self.time_passed = 0
        self.winner = 'prey'

        if role == 'pred':
            return self.pred_observations
        else:
            return self.prey_observations

    def pred_step(self, action_index, epsilon=0, mode='learning'):
        """
        Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)

        :param action_index: Index of the action that should be executed.
        :param epsilon: Current epsilon (for epsilon greedy; used here just for printing)
        :return: tuple of observed state, observed reward, wheter the episode is done and information (not used here)
        """
        # ----- Performing action
        old_obs = self.pred_observations[:5]
            
        action = self.actions[action_index]
        # perform action in environment
        self.pred.move(action[0], action[1], action[2])
        # get camera observations
        self.pred_observations = self.get_camera_observations()
        # append time passed from respective action
        self.time_passed += action[2]

        # ------ Calculating reward
        reward = self.get_pred_reward()
        self.accu_reward += reward

        # ------ printing for debugging
        print('\n---- PREDATOR ----')
        print('action:', action_index)
        print('reward:', reward)
        # print('elapsed time:', self.time_passed)

        # ------ Stopping and resetting
        # reset every 30s or as soon as prey is caught

        done = False
        if mode == 'learning':
            if self.time_passed >= 30000:
                done = True
                print('! TIME PASSED !')
            # 20 = reward for eating prey, change if reward changes
            if reward >= 20:
                self.winner = 'pred'
                done = True
                print('! PREY CAUGHT !')

        # reset metrics after each episode
        if done:
            print('episode done')
            self.episode_counter += 1
            # save the normalized time in dataframe
            entry = {
                'episode_index': self.episode_counter, 
                'time_passed': self.time_passed,
                'accu_reward': self.accu_reward,
                'winner': self.winner}
            self.df = self.df.append(entry, ignore_index=True)

            # write dataframe to disk
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/pred_{mode}_progress.tsv', sep='\t',
                           mode='w+')

            # move back
            self.pred.move(-20, -20, 2000)
            # sleep for 10 seconds emulating reset pos
            self.pred.sleep(4)

        self.pred_observations = self.pred_observations + old_obs + [(action_index+1) / len(self.actions)]

        return self.pred_observations, reward, done, {}

    def prey_step(self, action_index, epsilon=0, mode='learning'):
        """
        Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)

        :param action_index: Index of the action that should be executed.
        :param epsilon: Current epsilon (for epsilon greedy; used here just for printing)
        :return: tuple of observed state, observed reward, wheter the episode is done and information (not used here)
        """
        # ----- Performing action
        action = self.actions[action_index]
        # perform action in environment
        self.prey.move(action[0], action[1], action[2])
        # save stopping ir readings for relevant sensors
        self.prey_observations = self._get_sensor_observations()
        # append time passed from respective action
        self.time_passed += action[2]
        
        # ------ Calculating reward
        reward = self._compute_sensor_penalty()
        self.accu_reward += reward
        print('\n---- PREY ----')
        print('action:', action_index)
        print('reward:', reward)

        # if time passed supersedes threshold, stop episode
        done = False
        if mode == 'learning':
            if self.time_passed >= 30000:
                done = True
                print('! TIME PASSED !')

        # ------ Getting validation metrics
        if done:
            print('episode done')
            self.episode_counter += 1
            # save the normalized time in dataframe
            entry = {
                'episode_index': self.episode_counter, 
                'time_passed': self.time_passed,
                'accu_reward': self.accu_reward,
                }
            self.df = self.df.append(entry, ignore_index=True)

            # write dataframe to dataframe
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/prey_{mode}_progress.tsv', sep='\t', mode='w+')

        return self.prey_observations, reward, done, {}

    def _compute_sensor_penalty(self):
        errors = np.array([1.5 / math.sqrt(self.prey_observations[i]) for i in range(len(self.prey_observations))])
        sum = errors.sum() / 4
        return -sum + 4

    def get_camera_observations(self):
        image = self.pred.get_image_front()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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

        self.img = image
        self.mask = mask

        return cam_obs

    def _get_sensor_observations(self, pred=False):
        '''
        Reads sensor information and returns normalized an thresholded values.
        '''
        # all sensors: backR, backC, backL, frontRR, frontR, frontC, frontL, frontLL]
        # reading only sensors: backC, frontRR,  frontC, frontLL
        if pred:
            observation = [self.pred.read_irs()[i] for i in [3, 5, 7]]
        else:
            observation = [self.prey.read_irs()[i] for i in [1, 3, 5, 7]]
        observation = [0.15 if observation[i] == False else observation[i] for i in
                       range(len(observation))]  # false -> 0.15
        # we need to introduce a threshold s.t. only distances below 0.15 are counted. Otherwise the distances
        # will OFTEN be triggered when just moving in an empy plane because of the tilt of the robot.
        observation = [0.15 if observation[i] > 0.15 else observation[i] for i in range(len(observation))]
        return observation

    def get_pred_reward(self):
        cam_obs = self.pred_observations
        ir_obs = self._get_sensor_observations(pred=True)[1:]

        # object is close to predator
        ir_close = False
        for v in ir_obs:
            if v < 0.05:
                ir_close = True
                break
        # prey is visible in camera
        cam_close = False
        for v in cam_obs:
            if v > 0.25:
                cam_close = True
                break

        # if an object is object sensed and prey is visible
        if ir_close and cam_close:
            print(ir_obs)
            print(cam_close)
            reward = 20
        else:
            reward = sum(cam_obs)/5

        return reward
