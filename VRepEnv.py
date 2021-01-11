class VRepEnv:
    """Class to plug the VRep simulator environment into the Stable - baseline DQN algorithm."""

    def __init__(self, action_space):
        self.action_space = action_space  # might have to be a Gym Action Space

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        """Performs the action in the environment and returns the new observations (state), reward, done (?) and info(?)"""
        raise NotImplementedError()