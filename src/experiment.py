from VRepEnv import VRepEnv
from OurDQN import OurDQN
from OurMPLPolicy import OurMlpPolicy
import info

env = VRepEnv(info.actions, 5)
model = OurDQN(OurMlpPolicy, env)
model.learn(total_timesteps=25000, model_saving_path=info.model_save_file)
