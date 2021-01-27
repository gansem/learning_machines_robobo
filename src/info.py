import os

client = ''
ip = '127.0.0.1'

prey_actions = [
    (40, 40, 100),
    (-15, 15, 100),
    (15, -15, 100),
    (-40, -40, 100)]

pred_actions = [
    (40, 40, 100),
    (-15, 15, 100),
    (15, -15, 100),
    (20, 40, 100),
    (40, 20, 100)]

# task 1
# prey_actions = [(30, 30, 1000),    #straight forward
#            (10, -10, 500),   #spin right
#            (-10, 10, 500)]   #spin left

# actions = [(80, 80, 50, 1),    #straight forward
#            (30, -30, 50, 1),   #spin right
#            (-30, 30, 50, 1) ,   #spin left
#            (40, 80 , 50, 1),
#            (80, 40, 50, 1),
#            (60, 80, 50, 1),
#            (80, 60, 50, 1)]

# ---

mode = 'evaluating'
task = 'chasing_prey'
user = 'andi'
take = 'take_02'
scene = 'predator_prey_arena'
model_save_file = f'./results/{task}/{user}/{take}/{scene}'
pred_model_load_file = f'./results/{task}/{user}/{take}/pred_model'
prey_model_load_file = f'./results/{task}/{user}/{take}/prey_model'

# create folders if they do not exist
if not os.path.exists(f'./results/{task}'):
    os.makedirs(f'./results/{task}')
if not os.path.exists(f'./results/{task}/{user}'):
    os.makedirs(f'./results/{task}/{user}')
if not os.path.exists(f'./results/{task}/{user}/{take}'):
    os.makedirs(f'./results/{task}/{user}/{take}')
if not os.path.exists(f'./results/{task}/{user}/{take}/{mode}'):
    os.makedirs(f'./results/{task}/{user}/{take}/{mode}')
if not os.path.exists(f'./results/{task}/{user}/{take}/{mode}/pred_model'):
    os.makedirs(f'./results/{task}/{user}/{take}/{mode}/pred_model')
if not os.path.exists(f'./results/{task}/{user}/{take}/{mode}/prey_model'):
    os.makedirs(f'./results/{task}/{user}/{take}/{mode}/prey_model')