from stable_baselines.deepq.policies import FeedForwardPolicy


class OurMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[5, 5],
                 reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(OurMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, layers=layers,
                                        feature_extraction="mlp", obs_phs=obs_phs, dueling=dueling,
                                        layer_norm=False, **_kwargs)