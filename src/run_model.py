from VRepEnv import VRepEnv
from OurDQN import OurDQN
from OurMPLPolicy import OurMlpPolicy

actions = [(30, 30, 500, 1),    #straight forward
           (10, -10, 500, 1),   #spin right
           (-10, 10, 500, 1),   #spin left
           (-25, -25, 300, -1)] #straight backwards
env = VRepEnv(actions, 4)
model_to_load = 'learning_1_maze7'

# load model
model = OurDQN(OurMlpPolicy, env)
model.load(model_to_load)

# Do 1000 steps according to the model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)