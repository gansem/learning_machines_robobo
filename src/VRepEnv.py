import robobo
from gym.spaces import Discrete, Box
import math
import numpy as np
import pandas as pd
import info
import cv2
import vrep


class VRepEnv:
    """Class to plug the VRep simulator environment into the Stable - baseline DQN algorithm."""

    def __init__(self, actions, n_observations):
        """
        :param actions: list of actions. Each action is a four-tuple (left_speed, right_speed, duration, direction(-1=backwards, 1=forward))
        :param n_observations: number of sensors
        """
        self.rob = robobo.SimulationRobobo(info.client).connect(address='127.0.0.1', port=19997)
        # using action and observation spaces of Gym to minimize code alterations.
        self.actions = actions
        self.action_space = Discrete(len(actions))
        self.rob.play_simulation()
        self.observations = self.get_camera_observations()
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))
        self.time_passed = 0
        self.df = pd.DataFrame()
        self.accu_reward = 0
        self.episode_counter = 0

    def reset(self):
        '''
        Done at every episode end
        '''
        # validation measures
        self.accu_reward = 0
        self.time_passed = 0
        # TODO: random reset of food blocks

        return self.observations

    def step(self, action_index, epsilon=0):
        """
        Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)

        :param action_index: Index of the action that should be executed.
        :param task: which task is currently performed [1, 2, 3]
        :param epsilon: Current epsilon (for epsilon greedy; used here just for printing)
        :return: tuple of observed state, observed reward, wheter the episode is done and information (not used here)
        """
        # ----- Performing action
        action = self.actions[action_index]
        # perform action in environment
        self.rob.move(action[0], action[1], action[2])
        # save stopping ir readings for relevant sensors
        self.observations = self.get_camera_observations()
        self.time_passed += action[2]

        # ------ Calculating reward
        reward = self.get_reward()
        self.accu_reward += reward

        # ------ printing for debugging
        print('\n---- action:', action_index)
        print('reward:', reward)
        print('elapsed time:', self.time_passed)

        # ------ Stopping and resetting
        done = False
        #Todo: implement stopping condition
        #dummy function
        if self.time_passed > 3000:
            done = True

        # reset metrics after each episode
        if done:
            # save the normalized time in dataframe
            entry = {'avg_food_distance': self.get_avg_food_distance(),
                     'time_passed': self.time_passed,
                     'accu_reward': self.accu_reward}
            self.df = self.df.append(entry, ignore_index=True)

            # write dataframe to disk
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/learning_progress.tsv', sep='\t',
                           mode='w+')

        return self.observations, reward, done, {}

    def get_camera_observations(self, include_sensor=False):
        image = self.rob.get_image_front()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        low = np.array([25, 52, 72])
        high = np.array([102, 255, 255])
        mask = cv2.inRange(hsv, low, high)

        left = mask[:, 0:25]
        mid_left = mask[:, 25:51]
        mid = mask[:, 51:77]
        mid_right = mask[:, 77:103]
        right = mask[:, 103:]

        cam_values = [left, mid_left, mid, mid_right, right]

        cam_obs = [(np.sum(value) / (value.shape[0] * value.shape[1]))/255 for value in cam_values]

        if include_sensor:
            front_sensor = [self._get_sensor_observations()[2]]
            observation = [front_sensor.append(cam_ob) for cam_ob in cam_obs]
        else:
            observation = cam_obs

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

    def get_reward(self):
        #todo: calculate reward for given observation here
        # dummy
        return 1
        raise NotImplementedError

    # def get_validation_metrics(self, task, action_index, reward, distance, epsilon):
    #     '''
    #     Creates an entry for a data frame with metrics for a specific task and returns it.
    #     :param task: Task for which to collect the metrics. One of [1, 2, 3]
    #     :param action_index: Index of the current action in the action space.
    #     :param reward: reward currently observed.
    #     :param distance: travelled distance in the current step.
    #     :param epsilon: current epsilon.
    #     :return: entry with metrics
    #     '''
    #     action = self.actions[action_index]
    #     if task == 1:
    #         return self._get_validation_metrics_task1(action, action_index, reward, distance, epsilon)
    #
    # def _get_validation_metrics_task1(self, action, action_index, reward, distance, epsilon):
    #     in_object_range = False
    #     # if any IRS detects an object, add to validity measure
    #     if any(self.rob.read_irs()):
    #         in_object_range = True
    #     # add maximum +reward
    #     # distance = time (ms) * speed (?)
    #     self.v_measure_calc_distance += (action[3] * np.mean([action[0], action[1]])) / 10
    #     # calculate the inverse sensor distance as a sum
    #     self.v_measure_sensor_distance = np.sum([(0.2 - x) for x in self.rob.read_irs()])
    #     self.accu_v_measure_sensor_distance += self.v_measure_sensor_distance
    #
    #     # write to dataframe
    #     entry = {
    #         "action_index": int(action_index),
    #         "episode_index": int(self.episode_counter),
    #         "observations:": self.observations,
    #         "reward": reward,
    #         "object_in_range": in_object_range,
    #         "v_measure_calc_distance": self.v_measure_calc_distance,
    #         "v_measure_sensor_distance": self.v_measure_sensor_distance,
    #         "v_distance_reward": distance,
    #         "accu_v_measure_sensor_distance": self.accu_v_measure_sensor_distance,
    #         "accu_reward": self.accu_reward,
    #         "epsilon": epsilon
    #     }
    #     print(f"\n-- action_index: {action_index} --\nobservations: {self.observations} \nreward: {reward} \n"
    #           f"object in range: {in_object_range} \nv_measure_calc_distance: {self.v_measure_calc_distance}, \n"
    #           f"v_measure_sensor_distance: {self.v_measure_sensor_distance} \n"
    #           f"accu_v_measure_sensor_distance: {self.accu_v_measure_sensor_distance} \n"
    #           f"v_distance_reward: {distance} \nself.accu_reward: {self.accu_reward} \nepsilon: {epsilon}")
    #     return entry

    def get_avg_food_distance(self):
        '''
        Calculates the average distance of the food objects in the scene. Assuming that there are always 7 food objects.
        :return: Average distance of food objects
        '''
        food_names = ['Food', 'Food0', 'Food1', 'Food2', 'Food3', 'Food4', 'Food5']  # TOdO: make this generic for arbitraty foods
        food_positions = []
        # collect positions of all the foods
        for food in food_names:
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
