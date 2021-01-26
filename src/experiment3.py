from VRepEnv import VRepEnv
from OurDQN import OurDQN, OurDQNLearningThread
from OurMPLPolicy import OurMlpPolicy
import info
import pandas as pd
import robobo
import signal
import sys


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


signal.signal(signal.SIGINT, terminate_program)

# init VRep connections
rob = robobo.SimulationRobobo(info.client).connect(address=info.ip, port=19997)
rob.play_simulation()
prey = robobo.SimulationRoboboPrey().connect(address=info.ip, port=19989)
pred_env = VRepEnv(rob, info.pred_actions, 11, prey)
prey_env = VRepEnv(rob, info.prey_actions, 4, prey)

mode = 'learning'
if mode == 'learning':
    pred_model = OurDQN(OurMlpPolicy, pred_env, role='pred', policy_kwargs={'layers': [12, 8]})
    prey_model = OurDQN(OurMlpPolicy, prey_env, role='prey', policy_kwargs={'layers': [5, 5]})
    thread_pred = OurDQNLearningThread(pred_model, 25000, info.model_save_file+'_pred_')
    thread_prey = OurDQNLearningThread(prey_model, 50000, info.model_save_file+'_prey_')
    # model.learn(total_timesteps=25000, model_saving_path=info.model_save_file)
    thread_pred.start()
    thread_prey.start()
    # thread_pred.join()
    # thread_prey.join()
    print('done learning')

if mode == 'evaluating':
    i_eval = 2
    models_to_evaluate = [f'./results/foraging/seb/final_model/robobo_food_arena_2.92.model']#,f'E:/Uni/Learning_Machines/learning_machines_robobo/src/results/foraging/andi/take_01/food_arena_1.25.model']
    n_samples = 50
    results = pd.DataFrame()
    model_ind = 0
    for model in models_to_evaluate:
        model_ind += 1
        # reset data
        env.df = pd.DataFrame()
        model = OurDQN.load(model)
        for i in range(n_samples):
            obs = env.reset()
            done = False
            while not done:  # do an episode
                action, _states = model.predict(obs)
                obs, rewards, done, _info = env.step(action, mode=mode)
        result = pd.read_csv(f'./results/{info.task}/{info.user}/{info.take}/{mode}_progress.tsv', sep='\t')
        ind_col = [model_ind for i in range(n_samples)]
        result['Model_ind'] = ind_col
        results = results.append(result, ignore_index=True)
        results.to_csv(f'./results/{info.task}/evaluation/eval_{i_eval}.tsv', sep='\t', mode='w+')
