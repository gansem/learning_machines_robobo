from matplotlib import pyplot as plt
import pandas as pd

df_pred = pd.read_csv('E:/Uni/Learning_Machines/learning_machines_robobo/src/results/chasing_prey/andi/take_06/learning/pred_learning_progress.tsv', sep='\t')
summed_pred = [sum(df_pred['accu_reward'].iloc[0:i]) for i in range(len(df_pred))]
df_prey = pd.read_csv('E:/Uni/Learning_Machines/learning_machines_robobo/src/results/chasing_prey/andi/take_06/learning/prey_learning_progress.tsv', sep='\t')
summed_prey = [sum(df_prey['accu_reward'].iloc[0:i]) for i in range(len(df_prey))]
plt.plot(summed_pred, label='pred')
plt.plot(summed_prey, label='prey')
plt.legend()
plt.savefig('test.png')
plt.clf()
plt.plot(df_pred['accu_reward'], label='pred')
plt.plot(df_prey['accu_reward'], label='prey')
plt.ylim([-250, 250])
plt.legend()
plt.savefig('test2.png')
