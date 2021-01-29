from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def prop_prey_won(winner):
    pred_won = (winner == 'pred')
    sum = pred_won.sum()
    return sum / len(winner)

# ---- Calculate prop of Pred winning in evaluation
# df_normprey_fwdpred = pd.read_csv('./results/chasing_prey/andi/take_07/evaluating/pred_evaluating_progress.tsv', sep='\t')
# df_normprey_normpred = pd.read_csv('./results/chasing_prey/andi/take_08/evaluating/pred_evaluating_progress.tsv', sep='\t')
# df_fwdprey_fwdpred = pd.read_csv('./results/chasing_prey/et/take_10/pred_evaluating_progress.tsv', sep='\t')
# df_fwdprey_normpred = pd.read_csv('./results/chasing_prey/gd/take_10/pred_evaluating_progress.tsv', sep='\t')
# print('proportion prey winning:')
# prop = prop_prey_won(df_normprey_fwdpred['winner'])
# print('norm prey, fwd pred:', prop)
# prop = prop_prey_won(df_normprey_normpred['winner'])
# print('norm prey, norm pred:', prop)
# prop = prop_prey_won(df_fwdprey_fwdpred['winner'])
# print('fwd prey, fwd pred:', prop)
# prop = prop_prey_won(df_fwdprey_normpred['winner'])
# print('fwd prey, norm pred:', prop)

df_pred = pd.read_csv('C:/Users/Hammock/Documents/VU_masters_AI/learning_machines/learning_machines_robobo/src/results/chasing_prey/andi/take_05/learning/pred_learning_progress.tsv', sep='\t')
df_prey = pd.read_csv('C:/Users/Hammock/Documents/VU_masters_AI/learning_machines/learning_machines_robobo/src/results/chasing_prey/andi/take_05/learning/prey_learning_progress.tsv', sep='\t')
# -------- Plot rewards
summed_pred = [sum(df_pred['accu_reward'].iloc[0:i]) for i in range(len(df_pred))]
# compute moving average
avg_reward_pred = df_pred['accu_reward'].rolling(7)
avg_reward_pred = avg_reward_pred.mean()

summed_prey = [sum(df_prey['accu_reward'].iloc[0:i]) for i in range(len(df_prey))]
# compute moving average
avg_reward_prey = df_prey['accu_reward'].rolling(7)
avg_reward_prey = avg_reward_prey.mean()

plt.title('Accumulated reward per Episode (moving average)')
plt.plot(avg_reward_pred, label='pred')
plt.plot(avg_reward_prey, label='prey')
plt.xlabel('Episode')
plt.ylabel('Accumulated reward')
# plt.ylim([-100, 150])
plt.legend()
plt.savefig('./results/chasing_prey/ALL_learning_rewardsperepisode.eps')
plt.clf()

plt.title('Overall accumulated reward')
plt.plot(summed_pred, label='pred')
plt.plot(summed_prey, label='prey')
plt.xlabel('Episode')
plt.ylabel('Accumulated reward')
plt.legend()
plt.savefig('./results/chasing_prey/learning_totalrewards.eps')
plt.clf()

# ----- Plot times
summed_pred = [sum(df_pred['time_passed'].iloc[0:i]) for i in range(len(df_pred))]
# compute moving average
avg_time_pred = df_pred['time_passed'].rolling(7)
avg_time_pred = avg_time_pred.mean()

plt.title('passed time per Episode (moving average)')
plt.plot(avg_time_pred, label='pred')
plt.xlabel('Episode')
plt.ylabel('time passed')
plt.legend()
plt.savefig('./results/chasing_prey/learning_timessperepisode.eps')
plt.clf()

plt.title('Overall time passed')
plt.plot(summed_pred, label='pred')
plt.xlabel('Episode')
plt.ylabel('overall time passed')
plt.legend()
plt.savefig('./results/chasing_prey/learning_totaltime.eps')
plt.clf()
