import robobo
from gym.spaces import Discrete, Box
import math
import numpy as np
import pandas as pd
import info
import cv2


class VRepEnv:
    """Class to plug the VRep simulator environment into the Stable - baseline DQN algorithm."""

    def __init__(self, actions, n_observations, task=1):
        """
        :param actions: list of actions. Each action is a four-tuple (left_speed, right_speed, duration, direction(-1=backwards, 1=forward))
        :param n_observations: number of sensors
        :param task: which task is currently performed [1, 2, 3]
        """
        self.rob = robobo.SimulationRobobo(info.client).connect(address='127.0.0.1', port=19997)
        # using action and observation spaces of Gym to minimize code alterations.
        self.task = task
        self.actions = actions
        self.action_space = Discrete(len(actions))
        self.rob.play_simulation()
        self.observations = self.get_observations()
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))
        self.time_per_episode = 20000  # 20 seconds
        self.time_passed = 0
        self.df = pd.DataFrame()
        self.v_measure_calc_distance = 0
        self.v_measure_sensor_distance = 0
        self.accu_reward = 0
        self.accu_v_measure_sensor_distance = 0
        self.episode_counter = 0

        # temp
        self.img = []
        self.mask = []


    def reset(self):
        '''
        Done at every episode end
        '''
        self.time_passed = 0
        print('episode done')

        return self.observations

    def step(self, action_index, epsilon=0):
        """
        Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)

        :param action_index: Index of the action that should be executed.
        :param epsilon: Current epsilon (for epsilon greedy; used here just for printing)
        :return: tuple of observed state, observed reward, wheter the episode is done and information (not used here)
        """
        # ----- Performing action
        action = self.actions[action_index]
        # save starting position
        start_position = self.rob.position()
        # perform action in environment
        self.rob.move(action[0], action[1], action[2])
        # save stopping position
        stop_position = self.rob.position()
        # get readings for relevant sensors (IR or camera)
        self.observations = self.get_observations()
        # cv2.imwrite('test_img.png', imgs[0])
        # cv2.imwrite('test_mask.png', imgs[1])

        # cv2.imshow("image", image)
        # cv2.waitKey(0)

        # ------ Calculating reward
        distance = math.sqrt((stop_position[0] - start_position[0])**2 + (stop_position[1] - start_position[1])**2)
        reward = self.get_reward(action, distance)
        self.accu_reward += reward

        # ------ Getting validation metrics
        validation_metrics = self.get_validation_metrics(action_index, reward, distance, epsilon)
        # write to dataframe
        self.df = self.df.append(validation_metrics, ignore_index=True)

        # if time passed supersedes threshold, stop episode
        done = False
        self.time_passed += action[2]
        if self.time_passed >= self.time_per_episode:
            done = True

        # reset metrics after each episode
        if done:
            # write dataframe to disk
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/learning_progress.tsv', sep='\t', mode='w+')
            
            # validation measures
            self.v_measure_calc_distance = 0
            self.accu_v_measure_sensor_distance = 0
            self.accu_reward = 0
            self.episode_counter += 1

        return self.observations, reward, done, {}

    def get_rob_position(self):
        return self.rob.position()

    def get_observations(self):
        if self.task == 1:
            return self.get_sensor_observations()
        if self.task == 2:
            return self.get_camera_observations()

    def get_camera_observations(self):
        image = self.rob.get_image_front()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        low = np.array([25, 52, 72])
        high = np.array([102, 255, 255])
        mask = cv2.inRange(hsv, low, high)

        far_left = mask[:, 0:25]
        # left = mask[:, 0:51]
        mid_left = mask[:, 25:51]
        mid = mask[:, 51:77]
        mid_right = mask[:, 77:103]
        # right = mask[:, 77:]
        far_right = mask[:, 103:]

        cam_values = [far_left, mid_left, mid, mid_right, far_right]
        # cam_values = [left, mid, right]

        cam_obs = [(np.sum(value) / (value.shape[0] * value.shape[1]))/255 for value in cam_values]

        # Uncomment section below to also use front sensor
        # front_sensor = [self.get_sensor_observations()[2]]
        # observation = [front_sensor.append(cam_ob) for cam_ob in cam_obs]

        # uncomment to use only camera
        observation = cam_obs

        self.img = image
        self.mask = mask

        return observation

    def get_sensor_observations(self):
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

    def _compute_sensor_penalty_task1(self):
        errors = np.array([1.5/math.sqrt(self.observations[i]) for i in range(len(self.observations))])

        # give more importance to the front sensors
        errors[0] = 0.5*errors[0]
        errors[2] = 1.5*errors[2]
        sum = errors.sum()/4
        return -sum + 4

    def get_reward(self, action, travelled_distance):
        if self.task==1:
            return self._get_reward_task1(action, travelled_distance)
        elif self.task==2:
            return self._get_reward_task2()

    def _get_reward_task1(self, action, travelled_distance):
        distance_reward = action[3] * travelled_distance

        # get bonus if going straight. otherwise it just lears to turn in a circle
        if action[0] == action[1] and action[0] > 0:
            distance_reward *= 40
        sensor_penalty = self._compute_sensor_penalty_task1()

        return distance_reward + sensor_penalty

    def _get_reward_task2(self):
        _obs = self.observations

        # find mid in observation list
        mid_index = math.floor(len(_obs)/2)
        # find the most food in image section
        max_index = _obs.index(max(_obs))

        # if max is in left or right, apply negative reward according to distance
        if mid_index != max_index:
            reward = _obs[max_index] -1
        # if max is in mid, apply positive reward according to distance
        else:
            reward = _obs[max_index]

        return reward
    
    def _compute_sensor_penalty_task2(self):
        return


    def get_validation_metrics(self, action_index, reward, distance, epsilon):
        '''
        Creates an entry for a data frame with metrics for a specific task and returns it.
        :param action_index: Index of the current action in the action space.
        :param reward: reward currently observed.
        :param distance: travelled distance in the current step.
        :param epsilon: current epsilon.
        :return: entry with metrics
        '''
        action = self.actions[action_index]
        if self.task == 1:
            return self._get_validation_metrics_task1(action, action_index, reward, distance, epsilon)
        if self.task == 2:
            return self._get_validation_metrics_task2(action, action_index, reward, distance, epsilon)

    def _get_validation_metrics_task1(self, action, action_index, reward, distance, epsilon):
        in_object_range = False
        # if any IRS detects an object, add to validity measure
        if any(self.rob.read_irs()):
            in_object_range = True
        # add maximum +reward
        # distance = time (ms) * speed (?)
        self.v_measure_calc_distance += (action[3] * np.mean([action[0], action[1]])) / 10
        # calculate the inverse sensor distance as a sum
        self.v_measure_sensor_distance = np.sum([(0.2 - x) for x in self.rob.read_irs()])
        self.accu_v_measure_sensor_distance += self.v_measure_sensor_distance

        # write to dataframe
        entry = {
            "action_index": int(action_index),
            "episode_index": int(self.episode_counter),
            "observations": self.observations,
            "time_elapsed": self.time_passed,
            "reward": reward,
            "object_in_range": in_object_range,
            "v_measure_calc_distance": self.v_measure_calc_distance,
            "v_measure_sensor_distance": self.v_measure_sensor_distance,
            "v_distance_reward": distance,
            "accu_v_measure_sensor_distance": self.accu_v_measure_sensor_distance,
            "accu_reward": self.accu_reward,
            "epsilon": epsilon,
        }
        print(f"\n-- action_index: {action_index} --\nobservations: {self.observations} \nreward: {reward} \n"
              f"object in range: {in_object_range} \n"
              f"time elapsed: {self.time_passed} \n"
              f"v_measure_calc_distance: {self.v_measure_calc_distance} \n"
              f"v_measure_sensor_distance: {self.v_measure_sensor_distance} \n"
              f"v_distance_reward: {distance} \n"
              f"accu_v_measure_sensor_distance: {self.accu_v_measure_sensor_distance} \n"
              f"self.accu_reward: {self.accu_reward} \n"
              f"epsilon: {epsilon}")
        return entry

    def _get_validation_metrics_task2(self, action, action_index, reward, distance, epsilon):
        # in_object_range = False
        # if any IRS detects an object, add to validity measure
        # if any(self.rob.read_irs()):
        #     in_object_range = True

        # write to dataframe
        entry = {
            "action_index": int(action_index),
            "episode_index": int(self.episode_counter),
            "observations": self.observations,
            "time_elapsed": self.time_passed,
            "reward": reward,
            # "object_in_range": in_object_range,
            # "v_measure_calc_distance": self.v_measure_calc_distance,
            # "v_measure_sensor_distance": self.v_measure_sensor_distance,
            "v_distance_reward": distance,
            # "accu_v_measure_sensor_distance": self.accu_v_measure_sensor_distance,
            # "accu_reward": self.accu_reward,
            "epsilon": epsilon,
        }
        print(f"\n-- action_index: {action_index} --\nobservations: {self.observations} \nreward: {reward} \n"
            #   f"object in range: {in_object_range} \n"
              f"time elapsed: {self.time_passed} \n"
              f"v_measure_calc_distance: {self.v_measure_calc_distance} \n"
            #   f"v_measure_sensor_distance: {self.v_measure_sensor_distance} \n"
              f"v_distance_reward: {distance} \n"
            #   f"accu_v_measure_sensor_distance: {self.accu_v_measure_sensor_distance} \n"
            #   f"self.accu_reward: {self.accu_reward} \n"
              f"epsilon: {epsilon}")
        return entry

