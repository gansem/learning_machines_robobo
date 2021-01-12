import robobo
from gym.spaces import Discrete, Box
import math
import numpy as np

class VRepEnv:
    """Class to plug the VRep simulator environment into the Stable - baseline DQN algorithm."""

    def __init__(self, actions, n_observations):
        """

        :param actions: list of actions. Each action is a four-tuple (left_speed, right_speed, duration, direction(-1=backwards, 1=forward))
        :param n_observations: number of sensors
        """
        self.rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        # using action and observation spaces of Gym to minimize code alterations.
        self.actions = actions
        self.action_space = Discrete(len(actions))
        self.rob.play_simulation()
        self.observations = self.get_sensor_observations()
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))  # low and high depend on sensor values + normalization. Need to adjust.
        self.time_per_episode = 20000  # 20 seconds
        self.time_passed = 0

    def reset(self):
        self.rob.stop_world()
        self.time_passed = 0
        self.rob.play_simulation()
        # try:
        #     self.observations = self.get_sensor_observations()
        #     print('got observations', self.observations)
        # except VrepApiError as e:
        #     print(e)
        #     print('got observations', self.observations)
        #     self.reset()
        return self.observations


    def step(self, action_index):
        """Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)"""
        action = self.actions[action_index]
        # save starting position
        start_position = self.rob.position()
        # perform action in environment
        self.rob.move(action[0], action[1], action[2])
        # save stopping position
        stop_position = self.rob.position()
        # save stopping ir readings for relevant sensors
        self.observations = self.get_sensor_observations()

        # calculate distance reward with euclidean distance. Negative if action is going backwards
        distance_reward = action[3]*math.sqrt((stop_position[0] - start_position[0])**2
                                              + (stop_position[1] - start_position[1])**2)
        # get bonus if going straight
        if action[0] == action[1]:
            distance_reward += 0.5
        sensor_penalty = self._compute_sensor_penalty2()
        alpha = 6.0  # TODO: figure out proper scalars
        beta = 1.0
        overall_reward = alpha * distance_reward + beta * sensor_penalty
        print(overall_reward)

        # if time passed supersedes threshold, stop episode
        stop_episode = False
        self.time_passed += action[2]
        if self.time_passed > self.time_per_episode:
            stop_episode = True

        return self.observations, overall_reward, stop_episode, {}  # TODO: figure out last one params

    def get_rob_position(self):
        return self.rob.position()

    def get_sensor_observations(self):
        '''Reads sensor information and returns it. If distance too big and sensor gives False then this value becomes 1.
        We might want to introduce a normalization step here.
        '''
        observation = self.rob.read_irs()
        observation = [observation[i] for i in [1, 3, 5, 7]]  # reading only sensors: backC, frontRR,  frontC, frontLL
        observation = [1 if observation[i] == False else observation[i] for i in range(len(observation))]  # false -> 1
        return observation

    def _compute_sensor_penalty2(self):
        errors = np.array([1-self.observations[i] for i in range(len(self.observations))])
        # give bonus if no sensor is triggered
        if errors.sum() == 0:
            return 0.5
        # give more importance to the front sensors
        errors[0] = 0.5*errors[0]
        errors[2] = 1.5*errors[2]
        sum = errors.sum()
        return -sum

    def _compute_sensor_penalty(self, action):
        # action[0] is left wheel speed action [1] is right wheel speed.
        action_sum = action[0] + action[1]
        if action[0] < 0 and action[0] == action[1]:  # going backwards
            # - backC
            return -1*self.observations[0]
        elif action[0] == action[1]:  # going straight-forward
            # - frontC
            return -1*self.observations[2]

        # turning on point
        elif action[0] < 0:  # turning left on the spot
            return -1*self.observations[3]
        elif action[1] < 0:  # turning right on the spot
            return -1*self.observations[1]

        # for turning actions the relative part of the reward should depend on the speed in which the agent turned into
        # a 'bad' position. This is done with the weighted average of wheel speeds.
        elif action[0] > action[1]:  # going right
            # - (left/sum) * frontRR + (right/sum) * frontC
            return -1*((action[0]/action_sum)*self.observations[1]
                       + ((action[1]/action_sum)*self.observations[2]))
        elif action[0] < action[1]:  # going left
            # - (right/sum) * frontLL + (left/sum) * frontC
            return -1*((action[1]/action_sum)*self.observations[3]
                       + ((action[0]/action_sum)*self.observations[2]))

# Testing
# actions = [(50, 50, 1000, 1),
#            (20, 0, 1000, 1),
#            (0, 20, 1000, 1),
#            (-20, 20, 1000, 1),
#            (-20, -20, 1000, -1),
#            (60, 20, 1000, 1)]
# env = VRepEnv(actions, 4)
# import random
# for i in range(3):
#     print(env.step(0), 0)
# for i in range(20):
#     action = random.choice(range(6))
#     print(env.step(action), action)

# env.rob.stop_world()