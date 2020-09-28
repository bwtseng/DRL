import argparse
import gym
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from drawnow import drawnow
import matplotlib.pyplot as plt
from torch.autograd import Variable


last_score_plot = [-100]
avg_score_plot = [-100]


def draw_fig():
	plt.title('reward')
	plt.plot(last_score_plot, '-')
	plt.plot(avg_score_plot, 'r-')

"""
parser = argparse.ArgumentParser(description='PyTorch PPO solution of MountainCarContinuous-v0')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=1e-3)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--clip_epsilon', type=float, default=0.1)
parser.add_argument('--gae_lambda', type=float, default=0.995)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--max_episode', type=int, default=300)
cfg = parser.parse_args()

env = gym.make('MountainCarContinuous-v0')
"""

class running_state:
	def __init__(self, state):
		self.len = 1
		self.running_mean = state
		self.running_std = state ** 2

	def update(self, state):
		self.len += 1
		old_mean = self.running_mean.copy()
		self.running_mean[...] = old_mean + (state - old_mean) / self.len
		self.running_std[...] = self.running_std + (state - old_mean) * (state - self.running_mean)

	def mean(self):
		return self.running_mean

	def std(self):
		return np.sqrt(self.running_std / (self.len - 1))


class Actor(nn.Module):
	def __init__(self, train=False):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(2, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc_mean = nn.Linear(64, 1)
		self.fc_mean.weight.data.mul_(0.1)
		self.fc_mean.bias.data.mul_(0.0)
		self.train = train
		#self.fc_log_std = nn.Linear(64, 1)
		self.fc_log_std = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		#x = F.elu(self.fc1(x))
		#x = F.elu(self.fc2(x))
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		action_mean = self.fc_mean(x)
		#action_std = torch.exp(self.fc_log_std(x))
		if self.train:
			action_std = torch.exp(self.fc_log_std.expand_as(action_mean))
		else:
			action_std = self.fc_log_std.expand_as(action_mean)
		#action_std = self.fc_log_std.expand_as(action_mean)
		return action_mean.squeeze(), action_std.squeeze()


def get_action(state):
	action_mean, action_std = actor(state)
	action_dist = torch.distributions.Normal(action_mean, action_std)
	action = action_dist.sample()
	return action.item()


def synchronize_actors():
	for target_param, param in zip(actor_old.parameters(), actor.parameters()):
		target_param.data.copy_(param.data)


def update_actor(state, action, advantage, args):
	mean_old, std_old = actor_old(state)
	action_dist_old = torch.distributions.Normal(mean_old, std_old)
	action_log_probs_old = action_dist_old.log_prob(action)

	mean, std = actor(state)
	action_dist = torch.distributions.Normal(mean, std)
	action_log_probs = action_dist.log_prob(action)

	# update old actor before update current actor
	synchronize_actors()

	r_theta = torch.exp(action_log_probs - action_log_probs_old)
	surrogate1 = r_theta * advantage
	surrogate2 = torch.clamp(r_theta, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantage
	loss = -torch.min(surrogate1, surrogate2).mean()

	entropy = action_dist.entropy()
	loss = torch.mean(loss - 1e-2 * entropy)

	actor_optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm(actor.parameters(), 40)
	actor_optimizer.step()
	return


class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(2, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 1)
		self.fc3.weight.data.mul_(0.1)
		self.fc3.bias.data.mul_(0.0)


	def forward(self, x):
		#x = F.elu(self.fc1(x))
		#x = F.elu(self.fc2(x))
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))  
		value = self.fc3(x)
		return value.squeeze()


def get_state_value(state):
	state_value = critic(state)
	return state_value


def update_critic(state, target, args):
	state_value = critic(state)
	loss = F.mse_loss(state_value, target)
	critic_optimizer.zero_grad()
	loss.backward()
	critic_optimizer.step()
	return

"""
actor = Actor()
actor_old = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
"""

def train(args):
	state = env.reset()
	state_stat = running_state(state)

	for i in range(args.max_episode):
		start_time = time.perf_counter()
		episode_score = 0
		episode = 0
		memory = []

		with torch.no_grad():
			while len(memory) < args.batch_size:
				episode += 1
				state = env.reset()
				state_stat.update(state)
				state = np.clip((state - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)

				for s in range(1000):
					action = get_action(torch.tensor(state).float()[None, :])
					next_state, reward, done, _ = env.step([action])

					state_stat.update(next_state)
					next_state = np.clip((next_state - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)
					memory.append([state, action, reward, next_state, done])

					state = next_state
					episode_score += reward

					if done:
						break

			state_batch, \
			action_batch, \
			reward_batch, \
			next_state_batch, \
			done_batch = map(lambda x: np.array(x).astype(np.float32), zip(*memory))

			state_batch = torch.tensor(state_batch).float()
			values = get_state_value(state_batch).detach().cpu().numpy()

			returns = np.zeros(action_batch.shape)
			deltas = np.zeros(action_batch.shape)
			advantages = np.zeros(action_batch.shape)

			prev_return = 0
			prev_value = 0
			prev_advantage = 0
			for i in reversed(range(reward_batch.shape[0])):
				returns[i] = reward_batch[i] + args.gamma * prev_return * (1 - done_batch[i])
				# generalized advantage estimation
				deltas[i] = reward_batch[i] + args.gamma * prev_value * (1 - done_batch[i]) - values[i]
				advantages[i] = deltas[i] + args.gamma * args.gae_lambda * prev_advantage * (1 - done_batch[i])

				prev_return = returns[i]
				prev_value = values[i]
				prev_advantage = advantages[i]

		advantages = (advantages - advantages.mean()) /  (advantages.std() + 1e-4)

		advantages = torch.tensor(advantages).float()
		action_batch = torch.tensor(action_batch).float()
		returns = torch.tensor(returns).float()

		# using discounted reward as target q-value to update critic
		update_critic(state_batch, returns, args)

		update_actor(state_batch, action_batch, advantages, args)

		episode_score /= episode
		print('last_score {:5f}, steps {}, ({:2f} sec/eps)'.
			  format(episode_score, len(memory), time.perf_counter() - start_time))

		avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_score * 0.01)
		last_score_plot.append(episode_score)

		if episode % 8 == 0:
		  torch.save({"state_dict": actor.state_dict()}, "ppo_actor.tar.pth")
		  torch.save({"state_dict": critic.state_dict()}, "ppo_critic.tar.pth")
		#drawnow(draw_fig)

	env.close()


def test(args):
	#actor = Actor()
	#actor_old = Actor()
	"""
	class Actor(nn.Module):
	  def __init__(self):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(2, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc_mean = nn.Linear(64, 1)
		#self.fc_log_std = nn.Linear(64, 1)
		self.fc_log_std = nn.Parameter(torch.zeros(1))

	  def forward(self, x):
		#x = F.elu(self.fc1(x))
		#x = F.elu(self.fc2(x))
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		action_mean = self.fc_mean(x)
		#action_std = torch.exp(self.fc_log_std(x))
		#action_std = torch.exp(self.fc_log_std.expand_as(action_mean))
		action_std = self.fc_log_std.expand_as(action_mean)
		#action_std =
		return action_mean.squeeze(), action_std.squeeze()

	class Critic(nn.Module):
	  def __init__(self):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(2, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 1)

	  def forward(self, x):
		#x = F.elu(self.fc1(x))
		#x = F.elu(self.fc2(x))
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))  
		value = self.fc3(x)
		return value.squeeze()
	"""

	actor.load_state_dict(torch.load("/home/bwtseng/Downloads/DRL/ppo_actor.tar.pth")["state_dict"])
	critic.load_state_dict(torch.load("/home/bwtseng/Downloads/DRL/ppo_critic.tar.pth")["state_dict"])
	env = gym.make('MountainCarContinuous-v0')
	steps, rewards = [], []
	for episode in range(args.test_episode_num):
		S = env.reset()
		iter_cnt, total_reward = 0, 0
		while True:
			env.render()
			S = torch.FloatTensor(S)
			iter_cnt += 1
			mu, log_sigma = actor(Variable(S))
			action = torch.normal(mu, torch.exp(log_sigma))
			#A = agent.select_best_action(S)
			#S_p, R, is_done = env.take_one_step(action.item())
			S_p, R, is_done, _ = env.step([action.item()])
			total_reward += R
			S = S_p
			if is_done:
				total_reward = round(total_reward, 2)
				steps.append(iter_cnt)
				rewards.append(total_reward)
				print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
				break
	env.close()

if __name__ == '__main__':
	#main()
	parser = argparse.ArgumentParser(description='PyTorch PPO Training')
	parser.add_argument("--train", default=False, action="store_true")
	parser.add_argument("--test", default=False, action="store_true")
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--actor_lr', type=float, default=1e-3)
	parser.add_argument('--critic_lr', type=float, default=1e-3)
	parser.add_argument('--clip_epsilon', type=float, default=0.1)
	parser.add_argument('--gae_lambda', type=float, default=0.995)
	parser.add_argument('--batch_size', type=int, default=5000)
	parser.add_argument('--max_episode', type=int, default=300)
	parser.add_argument('--test_episode_num', type=int, default=10)
	args = parser.parse_args()

	if args.train:
		actor = Actor(train=True)
		actor_old = Actor(train=True)
		critic = Critic()
		actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
		critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
		env = gym.make('MountainCarContinuous-v0')
		#agent = DQN(env_shape)
		train(args)
	else:
		actor = Actor(train=False)
		critic = Critic()
		#actor_old = Actor()
		test(args)