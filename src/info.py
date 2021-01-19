import os

client = ''
ip = '127.0.0.1'


actions = [(30, 30, 1000, 1),    #straight forward
           (10, -10, 700, 1),   #spin right
           (-10, 10, 700, 1)]   #spin left
        #    (-25, -25, 300, -4)] #straight backwards

# actions = [
#     (35, 35, 500, 1),       # forward (0)
#     (10, 10, 500, 1),       # forward-slow (1)
#     (10, 35, 500, 0.8),        # turn left (2)
#     (35, 10, 500, 0.8),        # turn right (3)
#     (-10, 10, 500, 0.4),      # spin left (4)
#     (10, -10, 500, 0.4),      # spin right (5)
#     #(-20, -10, 500, -1),     # backwards--turn-left (5)
#     #(-10, -20, 500, -1),     # backwards--turn-right (6)
# ]

# ---

task = 'foraging'
user = 'et'
take = 'take_04'
scene = 'food_arena'
model_save_file = f'./results/{task}/{user}/{take}/{scene}'
model_load_file = f'./results/{task}/{user}/{take}/foraging_model'

# create folder if they do not exist
if not os.path.exists(f'./results/{task}'):
    os.makedirs(f'./results/{task}')
if not os.path.exists(f'./results/{task}/{user}'):
    os.makedirs(f'./results/{task}/{user}')
if not os.path.exists(f'./results/{task}/{user}/{take}'):
    os.makedirs(f'./results/{task}/{user}/{take}')
