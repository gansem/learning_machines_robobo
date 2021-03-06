import robobo
from gym.spaces import Discrete, Box
import math
import numpy as np
import pandas as pd
import info


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
        self.observations = self.get_sensor_observations()
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))
        self.time_per_episode = 20000  # 20 seconds
        self.time_passed = 0
        self.df = pd.DataFrame()
        self.v_measure_calc_distance = 0
        self.v_measure_sensor_distance = 0
        self.accu_reward = 0
        self.accu_v_measure_sensor_distance = 0
        self.episode_counter = 0

    def reset(self):
        '''
        Done at every episode end
        '''
        self.time_passed = 0
        print('episode done')

        return self.observations

    def step(self, action_index, task=1, epsilon=0, testing_model=False):
        """
        Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)

        :param action_index: Index of the action that should be executed.
        :param task: which task is currently performed [1, 2, 3]
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
        # save stopping ir readings for relevant sensors
        self.observations = self.get_sensor_observations()

        # ------ Calculating reward
        distance = math.sqrt((stop_position[0] - start_position[0])**2 + (stop_position[1] - start_position[1])**2)
        reward = self.get_reward(action, distance, task)
        self.accu_reward += reward

        # ------ Getting validation metrics
        validation_metrics = self.get_validation_metrics(task, action_index, reward, distance, epsilon)
        # write to dataframe
        self.df = self.df.append(validation_metrics, ignore_index=True)

        # if time passed supersedes threshold, stop episode
        done = False
        self.time_passed += action[2]
        if self.time_passed >= self.time_per_episode:
            done = True

        # reset metrics after each episode
        if done and not testing_model:
            # write dataframe to disk
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/learning_progress.tsv', sep='\t', mode='w+')
            
            # validation measures
            self.v_measure_calc_distance = 0
            self.accu_v_measure_sensor_distance = 0
            self.accu_reward = 0
            self.episode_counter += 1
        
        # write at every step
        if testing_model:
            self.df.to_csv(f'results/{info.task}/smart_robobo_0300h/{info.scene}/learning_progress.tsv', sep='\t', mode='w+')

        return self.observations, reward, done, {}

    def get_rob_position(self):
        return self.rob.position()

    def get_sensor_observations(self):
        '''
        Reads sensor information and returns normalized an thresholded values.
        '''
        observation = self.rob.read_irs()
        observation = [observation[i] for i in [1, 3, 5, 7]]  # reading only sensors: backC, frontRR,  frontC, frontLL
        observation = [0.15 if observation[i]==False else observation[i] for i in range(len(observation))]  # false -> 0.2
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

    def get_reward(self, action, travelled_distance, task):
        if task==1:
            return self._get_reward_task1(action, travelled_distance)

    def _get_reward_task1(self, action, travelled_distance):
        distance_reward = action[3] * travelled_distance

        # get bonus if going straight. otherwise it just lears to turn in a circle
        if action[0] == action[1] and action[0] > 0:
            distance_reward *= 40
        sensor_penalty = self._compute_sensor_penalty_task1()

        return distance_reward + sensor_penalty

    def get_validation_metrics(self, task, action_index, reward, distance, epsilon):
        '''
        Creates an entry for a data frame with metrics for a specific task and returns it.
        :param task: Task for which to collect the metrics. One of [1, 2, 3]
        :param action_index: Index of the current action in the action space.
        :param reward: reward currently observed.
        :param distance: travelled distance in the current step.
        :param epsilon: current epsilon.
        :return: entry with metrics
        '''
        action = self.actions[action_index]
        if task == 1:
            return self._get_validation_metrics_task1(action, action_index, reward, distance, epsilon)

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
            "observations:": self.observations,
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
              f"object in range: {in_object_range} \nv_measure_calc_distance: {self.v_measure_calc_distance}, \n"
              f"v_measure_sensor_distance: {self.v_measure_sensor_distance} \n"
              f"accu_v_measure_sensor_distance: {self.accu_v_measure_sensor_distance} \n"
              f"v_distance_reward: {distance} \nself.accu_reward: {self.accu_reward} \nepsilon: {epsilon}")
        return entry
