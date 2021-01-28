from VRepEnv import VRepEnv
from OurDQN import OurDQN, OurDQNLearningThread, OurDQNEvaluatingThread
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
pred_env = VRepEnv(rob, info.pred_actions, 9, prey)
prey_env = VRepEnv(rob, info.prey_actions, 4, prey)

# toggle mode from info.py

if info.mode == 'learning':
    pred_model = OurDQN(OurMlpPolicy, pred_env, role='pred', policy_kwargs={'layers': [12, 6]})
    prey_model = OurDQN(OurMlpPolicy, prey_env, role='prey', policy_kwargs={'layers': [5, 5]})
    # if you want to train already learned models
    # pred_model = OurDQN.load('./results/chasing_prey/andi/take_04/predator_prey_arena_pred__2.22.model')
    # prey_model = OurDQN.load('./results/chasing_prey/andi/take_04/predator_prey_arena_prey__3.06.model')
    thread_pred = OurDQNLearningThread(pred_model, 50000, info.model_save_file+'_pred_')
    thread_prey = OurDQNLearningThread(prey_model, 80000, info.model_save_file+'_prey_')
    # model.learn(total_timesteps=25000, model_saving_path=info.model_save_file)
    thread_pred.start()
    thread_prey.start()
    # thread_pred.join()
    # thread_prey.join()
    print('done learning')

if info.mode == 'evaluating':
    # thread_pred = OurDQNEvaluatingThread(pred_env, 'pred', info.pred_model_load_file+'predator_prey_arena_pred__3.19.model')
    # thread_prey = OurDQNEvaluatingThread(prey_env, 'prey', info.prey_model_load_file+'predator_prey_arena_prey__6.94.model')
    thread_pred = OurDQNEvaluatingThread(pred_env, 'pred', 'C:/Users/Hammock/Documents/VU_masters_AI/learning_machines/learning_machines_robobo/results/chasing_prey/andi/take_04/learning/predator_prey_arena_pred__6.94.model')
    thread_prey = OurDQNEvaluatingThread(prey_env, 'prey', 'C:/Users/Hammock/Documents/VU_masters_AI/learning_machines/learning_machines_robobo/results/chasing_prey/andi/take_04/learning/predator_prey_arena_prey__6.94.model')
    results = pd.DataFrame()

    thread_pred.start()
    thread_prey.start()


    # model_ind = 0
    # for model in models_to_evaluate:
    #     model_ind += 1
    #     # reset data
    #     env.df = pd.DataFrame()
    #     model = OurDQN.load(model)
    #     for i in range(n_samples):
    #         obs = env.reset()
    #         done = False
    #         while not done:  # do an episode
    #             action, _states = model.predict(obs)
    #             obs, rewards, done, _info = env.step(action, mode=mode)
    #     result = pd.read_csv(f'./results/{info.task}/{info.user}/{info.take}/{mode}_progress.tsv', sep='\t')
    #     ind_col = [model_ind for i in range(n_samples)]
    #     result['Model_ind'] = ind_col
    #     results = results.append(result, ignore_index=True)
    #     results.to_csv(f'./results/{info.task}/evaluation/eval_{i_eval}.tsv', sep='\t', mode='w+')
