import robobo
from gym.spaces import Discrete, Box
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))  # low and high depend on sensor values + normalization. Need to adjust.
        self.time_per_episode = 20000  # 20 seconds
        self.time_passed = 0
        self.df = pd.DataFrame()
        self.v_measure_calc_distance = 0
        self.v_measure_sensor_distance = 0
        self.accu_reward = 0
        self.accu_v_measure_sensor_distance = 0
        # self.v_distance_reward = 0
        self.episode_counter = 0

    def reset(self):
        "Done at every episode end"
        # self.rob.stop_world()
        self.time_passed = 0
        print('episode done')

        return self.observations

    def step(self, action_index):
        """Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)"""
        # -----Performin action
        action = self.actions[action_index]
        # save starting position
        start_position = self.rob.position()
        # perform action in environment
        self.rob.move(action[0], action[1], action[2])
        # save stopping position
        stop_position = self.rob.position()
        # save stopping ir readings for relevant sensors
        self.observations = self.get_sensor_observations()

        # ------ Getting validation metrics
        in_object_range = False
        # if any IRS detects an object, add to validity measure
        if any(self.rob.read_irs()):
            in_object_range = True
        # add maximum +reward
        # distance = time (ms) * speed (?)
        self.v_measure_calc_distance += (action[3] * np.mean([action[0], action[1]]))/10
        # calculate the inverse sensor distance as a sum
        self.v_measure_sensor_distance = np.sum([(0.2-x) for x in self.rob.read_irs()])
        self.accu_v_measure_sensor_distance += self.v_measure_sensor_distance

        # ------ Calculating reward
        # calculate distance reward with euclidean distance. Negative if action is going backwards
        distance = math.sqrt((stop_position[0] - start_position[0])**2
                                              + (stop_position[1] - start_position[1])**2)
        distance_reward = action[3]*distance
        alpha = 1.0  # TODO: figure out proper scalars
        beta = 1.0
        # get bonus if going straight. otherwise it just lears to turn in a circle
        if action[0] == action[1] and action[0] > 0:
            distance_reward *= 40
        sensor_penalty = self._compute_sensor_penalty()
        overall_reward = alpha * distance_reward + beta * sensor_penalty


        self.accu_reward += overall_reward
        # self.v_distance_reward += distance_reward

        # if time passed supersedes threshold, stop episode
        done = False
        self.time_passed += action[2]
        if self.time_passed >= self.time_per_episode:
            done = True

        # write to dataframe
        self.df = self.df.append({
            "action_index": int(action_index),
            "episode_index": int(self.episode_counter),
            "observations:": self.observations,
            "reward": overall_reward,
            "object_in_range": in_object_range,
            "v_measure_calc_distance": self.v_measure_calc_distance,
            "v_measure_sensor_distance": self.v_measure_sensor_distance,
            "v_distance_reward": distance,
            "accu_v_measure_sensor_distance": self.accu_v_measure_sensor_distance,
            "accu_reward": self.accu_reward
            }, ignore_index=True)
        print(f"\n-- action_index: {action_index} --\nobservations: {self.observations} \nreward: {overall_reward} \nobject in range: {in_object_range} \nv_measure_calc_distance: {self.v_measure_calc_distance}, \nv_measure_sensor_distance: {self.v_measure_sensor_distance} \naccu_v_measure_sensor_distance: {self.accu_v_measure_sensor_distance} \nv_distance_reward: {distance_reward} \nself.accu_reward: {self.accu_reward}")

        # plot learning at the end of episode
        if done:
            # write dataframe to disk
            self.df.to_csv(f'results/{info.task}/{info.user}/{info.take}/learning_progress.tsv', sep='\t')
            # self.plot_learning()  # TODO: this probably takes a lot of time from learning and we can do it after.
            
            # validation measures
            self.v_measure_calc_distance = 0
            self.accu_v_measure_sensor_distance = 0
            # self.accu_distance_reward = 0
            self.accu_reward = 0
            self.episode_counter += 1

        return self.observations, overall_reward, done, {}

    def get_rob_position(self):
        return self.rob.position()

    def get_sensor_observations(self):
        '''Reads sensor information and returns it. If distance too big and sensor gives False then this value becomes 1.
        We might want to introduce a normalization step here.
        '''
        observation = self.rob.read_irs()
        observation = [observation[i] for i in [1, 3, 5, 7]]  # reading only sensors: backC, frontRR,  frontC, frontLL
        observation = [0.15 if observation[i]==False else observation[i] for i in range(len(observation))]  # false -> 0.2
        # we need to introduce a threshold s.t. only distances below 0.15 are counted. Otherwise the distances
        # will OFTEN be triggered when just moving in an empy plane because of the tilt of the robot.
        observation = [0.15 if observation[i] > 0.15 else observation[i] for i in range(len(observation))]
        return observation

    def _compute_sensor_penalty(self):
        errors = np.array([1.5/math.sqrt(self.observations[i]) for i in range(len(self.observations))])
        # give bonus if no sensor is triggered
        #if errors.sum() == 0:
        #    return 0.5
        # give more importance to the front sensors
        errors[0] = 0.5*errors[0]
        errors[2] = 1.5*errors[2]
        sum = errors.sum()/4
        return -sum + 4

    def plot_learning(self, episode_index=0):
        # add only the rewards obtained when object was detected
        _df = self.df[self.df['object_in_range'] == 1]
        _df.reset_index(inplace=True)
        # melt
        melt = _df.melt(id_vars='index', value_vars=['reward', 'v_measure_calc_distance', 'v_measure_sensor_distance', 'accu_reward', 'accu_v_measure_sensor_distance'])

        # r = np.sum(self.accu_reward)
        # s = np.sum(self.accu_v_measure_sensor_distance)

        # plot
        plt.figure()
        sns.lineplot(data=melt, x='index', y='value', hue='variable')
        # sns.lineplot(x=[i for i in range(self.episode_counter)], y=[r,s])
        plt.savefig(f"results/{info.task}/{info.user}/{info.take}/scene_{info.scene}/learning_{episode_index}.png")
        plt.clf()
