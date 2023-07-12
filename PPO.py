import gymnasium as gym
import torch
import numpy as np
import os
import random
from torch import nn
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, d_observation, d_action, hid):
        super(Policy, self).__init__()
        self.action = nn.Sequential(
            nn.Linear(d_observation, hid),
            nn.ReLU(),
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Linear(64, d_action),
            nn.Softmax(dim=1),
        )
        self.critic = nn.Sequential(
            nn.Linear(d_observation, hid),
            nn.ReLU(),
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

class Memory:
    def __init__(self):
        self.observations:list = []
        self.actions:list = []
        self.rewards:list = []
        self.terminated:list = []
        self.old_logprob:list = []
    def clear(self):
        del self.observations[:]
        del self.actions[:]
        del self.rewards[:]
        del self.terminated[:]
        del self.old_logprob[:]
    def append(self, observation, action, reward, terminated):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminated.append(terminated)
    def append_logprob(self, old_logprob):
        self.old_logprob.append(old_logprob)

    def Observation(self):
        return torch.tensor(np.array(self.observations), dtype=torch.float32).to(device)
    def Action(self):
        return torch.tensor(self.actions, dtype=torch.int64).to(device)
    def Reward(self):
        return torch.tensor(self.rewards, dtype=torch.float32).to(device)
    def Terminated(self):
        return torch.tensor(self.terminated, dtype=torch.float32).to(device)
    def Old_logprob(self):
        return torch.tensor(self.old_logprob, dtype=torch.float32).to(device)

class Agent:
    def __init__(self, device, gamma=0.99, episilon=0.2):
        self.gamma = gamma
        self.policy = Policy(8, 4, 128).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.Adam(self.policy.action.parameters(), lr=1e-3)
        # self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=1e-3)
        self.memory = Memory()
        self.episilon = episilon

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        prob = self.policy.action(observation)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        logprob = dist.log_prob(action).squeeze()  # [1]
        self.memory.append_logprob(logprob)
        return action.item()

    def evaluate(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor]:
        prob = self.policy.action(observation)
        dist = torch.distributions.Categorical(prob)
        logprob = dist.log_prob(action).squeeze()  # [episode_length]
        value = self.policy.critic(observation).squeeze()  # [episode_length]
        return logprob, value
        

    def update(self, k_epoch=5):
        def normalize(x: torch.Tensor, episilon=1e-5) -> torch.Tensor:
            return (x - x.mean())/(x.std()+episilon)
        
        def discount(self, rewards: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
            discount_reward = []
            r_tmp = 0
            for i in range(rewards.shape[0]-1, -1, -1):
                if terminated[i] == 1:
                    r_tmp = 0
                r_tmp = rewards[i] + self.gamma * r_tmp
                discount_reward.insert(0, r_tmp)
            return torch.stack(discount_reward)
        
        discount_reward = discount(self, self.memory.Reward(), self.memory.Terminated())
        discount_reward = normalize(discount_reward)
        old_logprob = self.memory.Old_logprob()

        mean_loss = 0
        for _ in range(k_epoch):
            logprob, value = self.evaluate(self.memory.Observation(), self.memory.Action())
            advantage = discount_reward - value
            advantage = normalize(advantage)
            ratio = torch.exp(logprob - old_logprob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.episilon, 1+self.episilon) * advantage
            loss = -torch.min(surr1, surr2) + nn.MSELoss()(value, discount_reward)
            # loss = -advantage * logprob + nn.MSELoss()(value, discount_reward)
            # loss = -torch.min(surr1, surr2)
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_loss += loss.item()

            # critic_loss = nn.MSELoss()(value, discount_reward)
            # self.critic_optimizer.zero_grad()
            # critic_loss.backward()
            # self.critic_optimizer.step()

            print("\rEpoch: {}, Loss: {}".format(_, loss.item()), end="")
            
        self.memory.clear()
        return mean_loss/k_epoch

    def append(self, *kwargs):
        self.memory.append(*kwargs)

    def save(self, model_name):
        torch.save(self.policy.state_dict(), model_name)

if __name__ == "__main__":
    test = True
    if not test:
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        agent = Agent(device)
        writer = SummaryWriter("./logs/PPO-same-optim")
        Num_episode = 200
        episode_length = 1000
        for episode in range(Num_episode):
            print("\nEpisode: {}".format(episode))
            total_reward = 0
            rounds = 1
            observation, info = env.reset()
            for t in range(episode_length):
                action = agent.act(torch.tensor(observation)[None,:].to(device))
                new_observation, reward, terminated, truncated, info = env.step(action)
                agent.append(observation, action, reward, terminated)
                total_reward += reward
                observation = new_observation
                if terminated:
                    observation, info = env.reset()
                    rounds += 1
            agent.save("model.pth")
            print("\nEpisode: {}, Mean_Reward: {}, Rounds: {}".format(episode, total_reward/rounds, rounds))
            loss = agent.update()
            writer.add_scalar("Mean_Reward", total_reward/rounds, episode)
            writer.add_scalar("loss", loss, episode)
        env.close()

    #test
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = Agent(device)
    agent.policy.load_state_dict(torch.load("best_model.pth"))
    observation, info = env.reset()
    total_rewards = 0
    while True:
        # env.render()
        action = agent.act(torch.tensor(observation)[None,:].to(device))
        observation, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        if terminated or truncated:
            observation, info = env.reset()
            print("Total Reward: {}".format(total_rewards))
            total_rewards = 0
    env.close()



