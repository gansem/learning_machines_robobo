import robobo
from gym.spaces import Discrete, Box
import math

class VRepEnv:
    """Class to plug the VRep simulator environment into the Stable - baseline DQN algorithm."""

    def __init__(self, actions, n_observations):
        """

        :param actions: list of actions. Each action is a four-tuple (left_speed, right_speed, duration, direction(-1=backwards, 1=forward))
        :param n_observations: number of sensors
        """
        self.rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        # using action and observation spaces of Gym to minimize code alterations.
        self.action_space = Discrete(len(actions))
        self.observation_space = Box(low=0.0, high=1.0, shape=(n_observations,))  # low and high depend on sensor values + normalization. Need to adjust.
        self.running = False
        self.observations = self.get_sensor_observations()

    def start(self):
        self.rob.play_simulation()
        self.running = True

    def reset(self):
        raise NotImplementedError()
        self.running = False

    def step(self, action):
        """Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)"""
        # save starting position
        start_position = self.rob.position()
        # perform action in environment
        self.rob.move(action[0], action[1], action[3])
        # save stopping position
        stop_position = self.rob.position()
        # save stopping ir readings for relevant sensors
        self.observations = self.get_sensor_observations()

        # calculate distance reward with euclidean distance. Negative if action is going backwards
        distance_reward = action[3]*math.sqrt((stop_position[0] - start_position[0])**2
                                              + (stop_position[1] - start_position[1])**2)
        sensor_penalty = self._compute_sensor_penalty(action)
        alpha = beta = 1.0  # TODO: figure out proper scalars
        overall_reward = alpha * distance_reward - beta * sensor_penalty

        return self.observations, overall_reward, None, None  # TODO: figure out last to params

    def get_rob_position(self):
        return self.rob.position()

    def get_sensor_observations(self):
        observation = self.rob.read_irs()
        return [observation[i] for i in [1, 3, 5, 7]]  # reading only sensors: backC, frontRR,  frontC, frontLL

    def _compute_sensor_penalty(self, action):
        # TODO: figure out what to do with negative speeds.
        action_sum = action[0] + action[1]
        if action[0] < 0 and action[1] < 0:  # going backwards
            # - backC
            return -1*self.observations[0]
        elif action[0] == action[1]:  # going straight-forward
            # - frontC
            return -1*self.observations[2]

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
