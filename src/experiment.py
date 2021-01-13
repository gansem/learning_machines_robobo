from VRepEnv import VRepEnv
from OurDQN import OurDQN
from OurMPLPolicy import OurMlpPolicy

actions = [(30, 30, 500, 1),    #straight forward
           (10, -10, 500, 1),   #spin right
           (-10, 10, 500, 1)]#,   #spin left
           #(-25, -25, 300, -4)] #straight backwards
env = VRepEnv(actions, 4)

model = OurDQN(OurMlpPolicy, env)

model.learn(total_timesteps=25000)
model.save('obstacle_test1')
