import os

client = ''
ip = '127.0.0.1'

prey_actions = [(80, 80, 100), (-30, 30, 100), (30, -30, 100), (-80, -80, 100)]
pred_actions = [(80, 80, 100), (-30, 30, 100), (30, -30, 100), (40, 80, 100), (80, 40, 100), (80, 60, 100), (60, 80, 100)]
actions = [(30, 30, 1000),    #straight forward
           (10, -10, 500),   #spin right
           (-10, 10, 500)]   #spin left

# actions = [(80, 80, 50, 1),    #straight forward
#            (30, -30, 50, 1),   #spin right
#            (-30, 30, 50, 1) ,   #spin left
#            (40, 80 , 50, 1),
#            (80, 40, 50, 1),
#            (60, 80, 50, 1),
#            (80, 60, 50, 1)]

# ---

task = 'chasing_prey'
user = 'et'
take = 'take_01'
scene = 'predator_prey_arena'
model_save_file = f'./results/{task}/{user}/{take}/{scene}'
model_load_file = f'./results/{task}/{user}/{take}/chasing_prey_model'

# create folder if they do not exist
if not os.path.exists(f'./results/{task}'):
    os.makedirs(f'./results/{task}')
if not os.path.exists(f'./results/{task}/{user}'):
    os.makedirs(f'./results/{task}/{user}')
if not os.path.exists(f'./results/{task}/{user}/{take}'):
    os.makedirs(f'./results/{task}/{user}/{take}')
