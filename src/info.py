client = ''
ip = '127.0.0.1'

actions = [(30, 30, 500, 1),    #straight forward
           (10, -10, 500, 1),   #spin right
           (-10, 10, 500, 1)]   #spin left
        #    (-25, -25, 300, -4)] #straight backwards

# actions = [
#     (35, 35, 500, 1),       # forward (0)
#     (10, 35, 500, 0.8),        # turn left (1)
#     (35, 10, 500, 0.8),        # turn right (2)
#     (-10, 10, 500, 0.4),      # spin left (3)
#     (10, -10, 500, 0.4),      # spin right (4)
#     (-20, -10, 500, -1),     # backwards--turn-left (5)
#     (-10, -20, 500, -1),     # backwards--turn-right (6)
# ]

# ---

task = 'obstacle_avoidance'
user = 'et'
take = 'take_01'
scene = 'scene_05'
model_save_file = f'./results/{task}/{user}/{take}/{scene}'
model_load_file = f'./results/{task}/{user}/{take}/scene_5_0150.model'
