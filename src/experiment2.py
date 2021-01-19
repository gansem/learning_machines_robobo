from VRepEnv import VRepEnv
from OurDQN import OurDQN
from OurMPLPolicy import OurMlpPolicy
import info
import pandas as pd
import numpy as np

env = VRepEnv(info.actions, 5)
mode = 'evaluating'
if mode == 'learning':
    model = OurDQN(OurMlpPolicy, env)
    model.learn(total_timesteps=25000, model_saving_path=info.model_save_file)

if mode == 'evaluating':
    i_eval = 1
    models_to_evaluate = [f'E:/Uni/Learning_Machines/learning_machines_robobo/src/results/foraging/andi/take_01/food_arena_2.638888888888889.model']#,f'E:/Uni/Learning_Machines/learning_machines_robobo/src/results/foraging/andi/take_01/food_arena_1.25.model']
    n_samples = 4
    results = pd.DataFrame()
    model_ind = 0
    for model in models_to_evaluate:
        model_ind += 1
        # reset data
        env.df = pd.DataFrame()
        obs = env.reset()
        model = OurDQN.load(model)
        for i in range(n_samples):
            done = False
            while not done:  # do an episode
                action, _states = model.predict(obs)
                obs, rewards, done, _info = env.step(action, mode=mode)
        result = pd.read_csv(f'results/{info.task}/{info.user}/{info.take}/{mode}_progress.tsv', sep='\t')
        ind_col = [model_ind for i in range(n_samples)]
        result['Model_ind'] = ind_col
        results = results.append(result, ignore_index=True)
    results.to_csv(f'results/{info.task}/evaluation/eval_{i_eval}.tsv', sep='\t', mode='w+')
