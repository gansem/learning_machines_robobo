from VRepEnv import VRepEnv
from OurDQN import OurDQN
from OurMPLPolicy import OurMlpPolicy
import info

env = VRepEnv(info.actions, 4)

# load model
model = OurDQN.load(info.model_load_file)

# Do 1000 steps according to the model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)