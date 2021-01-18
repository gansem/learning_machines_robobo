from VRepEnv import VRepEnv
from OurDQN import OurDQN
from OurMPLPolicy import OurMlpPolicy
import info

actions = info.actions
env = VRepEnv(actions, 4)

# load model
model = OurDQN.load(info.model_load_file)

time_passed = 0

# Do 1000 steps according to the model
obs = env.reset()
while True:
    action_index, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action_index, testing_model=True)

    # add time
    time_passed += actions[action_index][2]
    # if 5 minutes passed, stop
    if time_passed > 240000:
        break