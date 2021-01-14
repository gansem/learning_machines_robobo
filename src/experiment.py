from VRepEnv import VRepEnv
from OurDQN import OurDQN
from OurMPLPolicy import OurMlpPolicy

actions = [(30, 30, 500, 1),    #straight forward
           (10, -10, 500, 1),   #spin right
           (-10, 10, 500, 1)]#,   #spin left
           #(-25, -25, 300, -4)] #straight backwards

time_step = 500

actions = [
    (35, 35, time_step, 1),       # forward (0)
    (10, 35, time_step, 0.8),        # turn left (1)
    (35, 10, time_step, 0.8),        # turn right (2)
    (-10, 10, time_step, 0.4),      # spin left (3)
    (10, -10, time_step, 0.4),      # spin right (4)
    (-20, -10, time_step, -1),     # backwards--turn-left (5)
    (-10, -20, time_step, -1),     # backwards--turn-right (6)
]


env = VRepEnv(actions, 4)
model_file = 'obstacle_test5'
model = OurDQN(OurMlpPolicy, env)
model.learn(total_timesteps=25000, model_saving_path=model_file)
