import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym 

def draw_fig():
    plt.title('reward')
    plt.plot(last_score_plot, '-')
    plt.plot(avg_score_plot, 'r-')

BATCH_SIZE = 32
E_GREEDY = 0.999
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
TRAIN_EPISODE_NUM = 200
TEST_EPISODE_NUM = 10
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50).to(device)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, 3).to(device)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, env_shape, learning_rate=0.01, reward_decay=0.9):
        self.eval_net, self.target_net = Net(), Net()

        self.action_n = 3
        self.state_n = 2
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = E_GREEDY
        self.env_shape = env_shape

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, self.state_n * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy, find the action with maximum value here.
            actions_value = self.eval_net.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_shape == 0 else action.reshape(self.env_shape)
        else:  # random
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_n]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.state_n:self.state_n+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.state_n+1:self.state_n+2]).to(device)
        b_s_next = torch.FloatTensor(b_memory[:, -self.state_n:]).to(device)

        # q_eval w.r.t the action in experience,  DDQN could be applied here, just compute the value from target critic:
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_next).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self):
        #torch.save(self.eval_net.state_dict(), 'model/DQN/eval_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))
        #torch.save(self.target_net.state_dict(), 'model/DQN/target_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))

        torch.save({"state_dict":self.eval_net.state_dict()}, 'dqn_target.tar.pth')
        torch.save({"state_dict":self.target_net.state_dict()}, 'dqn_critic.tar.pth')



    def load_model(self, model_name):
        self.eval_net.load_state_dict(torch.load(model_name)["state_dict"])


def newReward(obsesrvation, obsesrvation_):
    return abs(obsesrvation_[0] - (-0.5))


def train(agent):
    records = []
    for episode in range(TRAIN_EPISODE_NUM):
        # initial
        observation = env.reset()
        iter_cnt, total_reward = 0, 0
        while True:
            iter_cnt += 1
            # fresh env
            #env.render()
            # Agent choose action based on observation
            action = agent.choose_action(observation)
            # Agent take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            reward = newReward(observation, observation_)
            # Agent learn from this transition
            agent.store_transition(observation, action, reward, observation_)
            if agent.memory_counter > MEMORY_CAPACITY:
                agent.learn()

            # Acumulate reward
            total_reward += reward
            # Swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward = round(total_reward, 2)
                records.append((iter_cnt, total_reward))
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                break

    # end of game
    print('--------------------------------')
    print('Complete training')
    env.close()

    agent.save_model()
    print("save model")


def test(agent):#mothod, model_path):
    #load model
    #agent.load_model(torch.load("/home/bwtseng/Downloads/DRL/mountain-car-v0/model/DQN/eval_0.01_0.999_32.pkl"))
    agent.load_model("/home/bwtseng/Downloads/DRL/dqn_critic.tar.pth")
    env = gym.make('MountainCar-v0')
    steps, rewards = [], []
    for episode in range(TEST_EPISODE_NUM):
        # initial
        observation = env.reset()
        iter_cnt, total_reward = 0, 0
        while True:
            iter_cnt += 1

            # fresh env
            env.render()
            # RL choose action based on observation
            action = agent.choose_action(observation)
            #action = get_action(torch.tensor(observation).float()[None, :], 0.001)
            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            reward = newReward(observation, observation_)

            # accumulate reward
            total_reward += reward
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward = round(total_reward, 2)
                steps.append(iter_cnt)
                rewards.append(total_reward)
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                break

    # end of game
    print('--------------------------------')
    print('After {} episode,\nthe average step is {},\nthe average reward is {}.'.format(TEST_EPISODE_NUM, sum(steps)/len(steps), sum(rewards)/len(rewards)))
    env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DDQN Training')
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()

    if args.train:
        env = gym.make('MountainCar-v0')
        # To confirm it's discrete action spacce.
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  
        agent = DQN(env_shape)

    else:
        agent = DQN(env_shape)
        test(agent)