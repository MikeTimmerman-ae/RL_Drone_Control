import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


dir = 'policy-PPO-v12/files/'

PPO_entropy_loss_data = pd.read_csv(dir + 'PPO_entropy_loss.csv')
PPO_explained_variance_data = pd.read_csv(dir + 'PPO_explained_variance.csv')
PPO_value_loss_data = pd.read_csv(dir + 'PPO_value_loss.csv')

PPO_rew_mean_data = pd.read_csv(dir + 'PPO_rew_mean.csv')

PPO_deviation_data = pd.read_csv(dir + 'PPO_deviation.csv')
PPO_success_data = pd.read_csv(dir + 'PPO_success.csv')
PPO_time_out_data = pd.read_csv(dir + 'PPO_time_out.csv')


# Plot success rates
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(PPO_deviation_data['Step'], PPO_deviation_data['Value'])
ax1.grid()
ax1.set_title('Deviations per Episode')

ax2.plot(PPO_success_data['Step'], PPO_success_data['Value'])
ax2.grid()
ax2.set_title('Successes per Episode')

ax3.plot(PPO_time_out_data['Step'], PPO_time_out_data['Value'])
ax3.grid()
ax3.set_title('Time-outs per Episode')


# Plot training statistics
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(PPO_entropy_loss_data['Step'], PPO_entropy_loss_data['Value'])
ax1.grid()
ax1.set_title('Entropy Loss')

ax2.plot(PPO_explained_variance_data['Step'], PPO_explained_variance_data['Value'])
ax2.grid()
ax2.set_title('Explained Variance')

ax3.plot(PPO_value_loss_data['Step'], PPO_value_loss_data['Value'])
ax3.grid()
ax3.set_title('Value Loss')


# Plot mean reward per episode
fig3 = plt.figure()

plt.plot(PPO_rew_mean_data['Step'], PPO_rew_mean_data['Value'])
plt.grid()
plt.title('Mean Reward per Episode')

plt.show()
